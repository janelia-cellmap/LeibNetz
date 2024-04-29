from leibnetz.nodes import Node


class FactorCompressedConvNode(Node):
    # Inspired by LoRA fine tuning method

    # TODO: Do with codebook adaptive lookup
    # increase ranks steadily until optimum found
    def __init__(
        self,
        input_keys,
        output_keys,
        input_nc,
        output_nc,
        kernel_sizes,
        rank_A: int = 8,
        rank_B: int = 8,
        output_key_channels=None,
        activation="ReLU",
        padding="valid",
        residual=False,
        padding_mode="reflect",
        norm_layer=None,
        dropout_prob=None,
        identifier=None,
    ) -> None:
        super().__init__(input_keys, output_keys, identifier)
        # matrices A and B are multiplied to compute the full weight matrix W for the forward pass
        self.rank_A = rank_A
        self.rank_B = rank_B
        self.output_key_channels = output_key_channels
        self._type = __name__.split(".")[-1]
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.padding = padding
        self.residual = residual
        self.padding_mode = padding_mode
        self.norm_layer = norm_layer
        self.dropout_prob = dropout_prob
        self.model = ...  # TODO
        self.color = "#00FFF0"
        self._convolution_crop = None
