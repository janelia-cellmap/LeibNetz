from torch.nn import Module
# defines baseclass for all nodes in the network
class Node(Module):
    def __init__(self) -> None:
        super().__init__()
        self._inputs = {}
        self._outputs = {}
        self._id = None
        self.resolution = None
        self._edges = []

    def __call__(self, *args, **kwargs):
