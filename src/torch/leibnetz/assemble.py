import torch

import logging
logger = logging.getLogger(__name__)

# function for assembling a network from a series of nodes with given inputs (either tensors or other node instances)
def assemble(node_list):
    raise NotImplementedError
