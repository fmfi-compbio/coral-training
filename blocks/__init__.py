
from .bonito import BLOCKS as BONITO_BLOCKS
from .custom import BLOCKS as CUSTOM_BLOCKS
from .pool import BLOCKS as POOL_BLOCKS
from .decoder import BLOCKS as DECODER_BLOCKS
from .paper import BLOCKS as PAPER_BLOCKS

BLOCKS = {
    **BONITO_BLOCKS,
    **CUSTOM_BLOCKS,
    **POOL_BLOCKS,
    **DECODER_BLOCKS,
    **PAPER_BLOCKS,
}

def Block(type="default", **kwargs):
    return BLOCKS[type](**kwargs)
