from cellmap_segmentation_challenge import train
from upath import UPath

import leibnetz
from common import (
    spatial_dims,
    batch_size,
    random_seed,
    optimizer,
    shared_kwargs,
)

model_to_load = model_name = (
    UPath(__file__).stem.removeprefix("train_") + f"_{random_seed}"
)
model_kwargs = {
    "subnet_dict_list": [
        {
            "top_resolution": (128,) * spatial_dims,
            **shared_kwargs,
        },
        {
            "top_resolution": (32,) * spatial_dims,
            **shared_kwargs,
        },
        {
            "top_resolution": (8,) * spatial_dims,
            **shared_kwargs,
        },
    ]
}

# Build the model
model = leibnetz.build_scalenet(**model_kwargs)

# Get the arrays needed for training from the model
input_array_info = model.input_shapes
target_array_info = model.output_shapes


# Define the optimizer
optimizer = optimizer(model.parameters())  # optimizer to use for training

mini_batch_size = 8
gradient_accumulation_steps = batch_size // mini_batch_size
batch_size = mini_batch_size

if __name__ == "__main__":
    train(__file__)
