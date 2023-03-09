from fire import Fire

from innofw.utils.checkpoint_utils import (
    add_metadata2model as cli_add_metadata2model,
)

if __name__ == "__main__":
    Fire(cli_add_metadata2model)
