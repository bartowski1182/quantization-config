# Per-tensor layout maps for llama.cpp quants

A layout map is a llama-quantize `--tensor-type-file`: one `^tensor$=type` line per
quantizable tensor, so a quant of a given ftype places its bits per tensor instead of
following llama-quantize's built-in per-name heuristic. The maps shipped in the
`layouts/` folder of a GGUF release are produced by exactly the code in this repo.

This repo is an export from the (currently private) service that builds and uploads
those releases. It is a snapshot per generator key, not a development repo: each commit
is one state of the generator, tagged `key-<generator key>`, which is the key every
release's `layout.json` records. Issues are welcome; changes land in the service repo
first and arrive here as the next snapshot.

## What's here

`layout/`, the generator package, stdlib only, with no container and no network:

- `dryrun.py`: the `llama-quantize --dry-run` argv and its output parser
- `generate.py`: pins, the embedding rule, the body solve, the map text and metadata
- `solver.py`: the type search under a base-share floor and the block/type tables
- `shape.py`: a body-shape key over a model's tensor list
- `prior.json`: the frozen cross-model prior the solve is calibrated on

`llama.cpp/llama-quant.cpp`, the override the CPU image is built with: at the pinned
llama.cpp tag, `src/llama-quant.cpp` is replaced by this file before the build. It
matters for reproduction because the dry runs and the quantize itself run against that
binary, and because the shape pins (any weight whose first dimension is not a multiple of
256) take their type from that binary's own heuristic, so the map is byte-neutral there.

## The prior

`prior.json` is frozen and never hand-edited: it is produced by the measurement campaign
that fit it, and any change to it changes the generator key. It has three parts:

- `prior`, the sensitivity cells, keyed `context|kind|band`. The context is `attn`,
  `gdn`, `dense` or `-` (the pooled cells), the kind is the tensor's role
  (`attn_q`, `attn_k`, `attn_v`, `attn_qkv`, `attn_gate`, `attn_output`, `ffn_up`,
  `ffn_gate`, `ffn_down`, `ssm_alpha`, `ssm_beta`, `ssm_out`), and the band is where the
  tensor's block sits in the model: `first`, `early`, `mid`, `late`, `last`. A cell's
  value is that group's measured share of the damage.
- `kind_prior`, the same values pooled over the bands, used as the fallback when a cell
  is missing.
- `r`, the damage of each quantization type relative to `q2_k`, which is `1.0`:
  for example `q3_k` 0.3227, `q4_k` 0.0551, `q5_k` 0.0178, `q6_k` 0.0079, and `q8_0`,
  `f16` and `f32` are `0.0`. The solve spends bits by damage per byte, `r` against
  the cell.

The cells were fit from degrade-one measurements on Qwen3.5-0.8B and 4B, with the
`dense` cells fit from granite-3b and granite-8b.

## How a map is built

The service does this per quant:

1. `dryrun.dry_run_args("q8_0", src, out, None, [], threads)`, run in the image, then
   `dryrun.parse_inventory(text)`: the q8_0 inventory, every tensor with its shape,
   source type and byte count.
2. `generate.covered_names(<the imatrix GGUF's tensor names>)`: which tensors the
   imatrix has data for.
3. `generate.find_pins(inv_q8, covered, mtp_blocks=<the model's MTP block ids>)`: the
   pins, being tensors the imatrix does not cover, tensors whose first dimension is not a
   multiple of 256, and tiny body tensors. Pins are the first lines of the map, because
   in llama-quantize the first matching rule wins.
4. `dryrun.dry_run_args(<the llama-quantize ftype>, src, out, imatrix,
   pins.class_a_types(<the rung ftype>), threads)`, run in the image, then
   `parse_inventory` again: the ftype's own inventory, which is what the heuristic
   would have produced.
5. `generate.build_map(<the rung ftype, lower case>, inv, pins, model=..., imatrix=...)`.
   `result.text` is the map, written as `<file>.tensor-types.txt`; `result.info` is the
   record written as `<file>.layout.json`, including `generator_key`, `quantize_ftype`
   and `llama_cpp_version`.
6. The quantize run:

```
./llama-quantize --imatrix <imatrix.gguf> \
    [--tensor-type ffn=mxfp4] \
    [--tensor-type 'blk[.]<N>[.]=q4_0' once per MTP block N] \
    --tensor-type-file <map.tensor-types.txt> \
    <src.gguf> <dst.gguf> <FTYPE> <threads>
```

The two bracketed lines are conditional: `ffn=mxfp4` only for gpt-oss models, and the
`blk[.]N[.]` pins only for models that have MTP or NextN blocks.

The rung and the ftype are not always the same name. A rung is a minimum share of body
bytes held at the base type; an `_L` name is the tier-L rung of its base type, and
llama-quantize has no such ftype, so `Q4_K_L` is quantized as `Q4_K_M` and `Q6_K_L` as
`Q6_K` while the map is solved for the `_L` rung. Every other rung uses its own name.
The `--tensor-type-file` goes last on the command line, after any `--tensor-type`, so
those earlier rules win where they overlap. There are no
`--output-tensor-type` or `--token-embedding-type` flags under a map: the map places the
embedding tensors itself.

## Recreating a released file

A mapped release ships `layouts/<file>.tensor-types.txt` and `layouts/<file>.layout.json`
next to the GGUF, and the release README links them.

1. Read `layouts/<file>.layout.json`: it names the `generator_key` and the
   `llama_cpp_version` the file was built with.
2. Check out the tag `key-<generator_key>` in this repo. That is the exact prior and the
   exact rules the map came from.
3. Build llama.cpp at the tag in `llama_cpp_version`, with this repo's
   `llama.cpp/llama-quant.cpp` copied over its `src/llama-quant.cpp`.
4. The released `.tensor-types.txt` is the map, so running the quantize command line
   above with the release's imatrix and the release's source GGUF reproduces the file.
   To check the map itself as well, rerun steps 1 to 5 above against the same source and
   imatrix and diff the result against the released map: it should be identical.

## This export

Generator key: a69a855b91615d22

The method write-up is linked from every release's README.
