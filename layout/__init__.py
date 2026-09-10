"""Per-tensor layout maps for llama-quantize.

A layout map is a `--tensor-type-file`: one `^tensor$=type` line per quantizable
tensor, computed per model from its dry-run inventory and its imatrix's tensor
list, so a quant of a given ftype places its bits where a cross-model
sensitivity prior says they buy the most quality. The ftype's name is a rung: a
minimum share of body bytes held at the base type (S 0.90 / M 0.70 / L 0.50),
with no byte budget — the resulting bitrate is reported, not targeted.

  `dryrun`    the llama-quantize --dry-run argv and its output parser
  `generate`  pins, the embedding rule, the body solve, the map text and metadata
  `solver`    the type search under a base-share floor and the block/type tables
  `prior.json`  the frozen cross-model prior the solve is calibrated on

Stdlib only: nothing here runs a container or touches the network — the caller
runs the dry runs and writes the files.
"""


class LayoutError(Exception):
    """A layout map cannot be built from these inputs."""
