from torch.nn import Module


# defines baseclass for all edges in the network
class Edge(Module):
    def __init__(self) -> None:
        super().__init__()
        self._inputs = {}
        self._outputs = {}
