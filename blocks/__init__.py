from .decoder import BLOCKS as DECODER_BLOCKS
from .paper import BLOCKS as PAPER_BLOCKS

BLOCKS = {
    **DECODER_BLOCKS,
    **PAPER_BLOCKS,
}

def Block(type="default", **kwargs):
    return BLOCKS[type](**kwargs)
