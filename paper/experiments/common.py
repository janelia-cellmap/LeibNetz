import subprocess
import os
import torch
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import (
    RandomContrast,
    GaussianNoise,
    Binarize,
    NaNtoNum,
    Normalize,
)

from upath import UPath

# %% Set hyperparameters and other configurations
spatial_dims = 3  # number of spatial dimensions in the input data
learning_rate = 0.0001  # learning rate for the optimizer
max_grad_norm = 1  # maximum gradient norm for gradient clipping
batch_size = 32  # batch size for the dataloader
epochs = 1000  # number of epochs to train the model for
iterations_per_epoch = 1000  # number of iterations per epoch
random_seed = int(os.environ.get("RANDOM_SEED", 42))  # random seed for reproducibility

classes = ["mito", "nuc", "er", "cell"]  # List of classes
load_model = "latest"  # load the latest model or the best validation model

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/{model_name}_{epoch}.pth"
).path  # path to save the model checkpoints
datasplit_path = "datasplit.csv"  # path to the datasplit file

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y", "z"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
}
train_raw_value_transforms = T.Compose(
    [
        T.ToDtype(torch.float),
        Normalize(),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        T.RandomApply(
            [
                RandomContrast(),
                GaussianNoise(std=0.02),
            ]
        ),
    ],
)
target_value_transforms = T.Compose(
    [
        T.ToDtype(torch.float),
        Binarize(threshold=0.5),
    ],
)  # target value transforms

use_mutual_exclusion = True  # whether to use mutual exclusion during training
force_all_classes = (
    False  # whether to force all classes to be present in each training sample
)
validation_prob = 0.1
datasets = ["*"]
crops = ["*"]

# Set a limit to how long the validation can take
validation_time_limit = 60  # time limit in seconds for the validation step


def optimizer(params):
    """Optimizer to use for training."""
    return torch.optim.RAdam(
        params,
        lr=learning_rate,
        decoupled_weight_decay=True,
        weight_decay=0.00025,
    )


# Define the common model hyperparameters
shared_kwargs = {
    "output_nc": len(classes),
    "final_activation": "Identity",
    "base_nc": 12,
    "nc_increase_factor": 2,
    "num_final_convs": 3,
    "convs_per_level": 3,
    "residual": True,
    "norm_layer": "InstanceNorm",
}

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
repo_path = (
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
)
repo_name = repo_path.split("/")[-1]
print(f"Current Git hash for {repo_name}: {git_hash}")
