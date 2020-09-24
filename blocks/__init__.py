
from .bonito import BLOCKS as BONITO_BLOCKS
from .custom import BLOCKS as CUSTOM_BLOCKS
from .pool import BLOCKS as POOL_BLOCKS

BLOCKS = {
    **BONITO_BLOCKS,
    **CUSTOM_BLOCKS,
    **POOL_BLOCKS,
}

def Block(type="default", **kwargs):
    return BLOCKS[type](**kwargs)
