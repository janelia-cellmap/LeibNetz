import torch


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, input_shapes, output_shapes, name="ModelWrapper"):
        """
        Wrap a model that takes an array input, or list of array inputs, and returns an array output, or list of array outputs, such that it is interchangeable with a LeibNet.
        """
        super().__init__()
        self.model = model
        self.input_shapes = {
            key: {k: tuple([int(s) for s in v]) for k, v in val.items()}
            for key, val in input_shapes.items()
        }
        self.output_shapes = {
            key: {k: tuple([int(s) for s in v]) for k, v in val.items()}
            for key, val in output_shapes.items()
        }
        self.name = name

        self.input_keys = list(input_shapes.keys())
        self.output_keys = list(output_shapes.keys())

    def get_example_inputs(self, device: torch.device = None):
        # function for generating example inputs
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        inputs = {}
        for k, v in self.input_shapes.items():
            inputs[k] = torch.rand(
                (
                    1,
                    1,
                )
                + tuple([int(s) for s in v["shape"]])
            ).to(device)
        return inputs

    def to(self, device):
        self.model.to(device)
        return self

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs[key] for key in self.input_keys]
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        outputs = self.model(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs = {key: val for key, val in zip(self.output_keys, outputs)}

        return outputs
