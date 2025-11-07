from cellmap_segmentation_challenge import train
from upath import UPath

import leibnetz
from common import (
    spatial_dims,
    random_seed,
    optimizer,
    shared_kwargs,
)

model_to_load = model_name = (
    UPath(__file__).stem.removeprefix("train_") + f"_{random_seed}"
)

# Build the model
model = leibnetz.build_unet(
    top_resolution=(8,) * spatial_dims,
    **shared_kwargs,
)

# Get the arrays needed for training from the model
input_array_info = model.input_shapes
target_array_info = model.output_shapes

# Define the optimizer
optimizer = optimizer(model.parameters())  # optimizer to use for training

if __name__ == "__main__":
    train(__file__)
