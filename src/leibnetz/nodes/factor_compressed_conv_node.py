from leibnetz.nodes import Node

class FactorCompressedConvNode(Node):
    def __init__(
        self,
        input_keys,
        output_keys,
        input_nc,
        output_nc,
        kernel_sizes,
        output_key_channels=None,
        activation="ReLU",
        padding="valid",
        residual=False,
        padding_mode="reflect",
        norm_layer=None,
        dropout_prob=None,
        identifier=None,
    ) -> None:
    self.rank_A = ...
    self.rank_B = ...
    # matrices A and B are multiplied to compute the full weight matrix W for the forward pass