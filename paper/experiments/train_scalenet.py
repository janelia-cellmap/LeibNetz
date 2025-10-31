from cellmap_segmentation_challenge import train
from upath import UPath

import leibnetz
from common import (
    spatial_dims,
    learning_rate,
    max_grad_norm,
    batch_size,
    epochs,
    iterations_per_epoch,
    random_seed,
    classes,
    load_model,
    logs_save_path,
    model_save_path,
    datasplit_path,
    spatial_transforms,
    train_raw_value_transforms,
    target_value_transforms,
    use_mutual_exclusion,
    force_all_classes,
    validation_time_limit,
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

mini_batch_size = 16
gradient_accumulation_steps = batch_size // mini_batch_size
batch_size = mini_batch_size

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    train(__file__)
