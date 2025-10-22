# %%
# Import the necessary libraries
# pip install git+https://github.com/janelia-cellmap/cellmap-data.git
from cellmap_data import CellMapDataLoader, CellMapDataSplit

# pip install git+https://github.com/janelia-cellmap/cellmap-train.git
from cellmap_train.loss import BCELoss

# pip install git+https://github.com/janelia-cellmap/LeibNetz.git
import leibnetz

from tqdm import tqdm
import torch

classes = ["mito", "nuc", "er", "cell"]  # List of classes
batch_size = 16  # Batch size
steps = 1000  # Steps per epoch
epochs = 100  # Number of epochs

# Define the model hyperparameters
shared_kwargs = {
    "output_nc": len(classes),
    "final_activation": "Identity",
    "base_nc": 12,
    "nc_increase_factor": 3,
    "num_final_convs": 3,
}
model_kwargs = {
    "subnet_dict_list": [
        {
            "top_resolution": (128, 128, 128),
            **shared_kwargs,
        },
        {
            "top_resolution": (32, 32, 32),
            **shared_kwargs,
        },
        {
            "top_resolution": (8, 8, 8),
            **shared_kwargs,
        },
    ]
}

# Build the model
model = leibnetz.build_scale_net(**model_kwargs)

# Get the arrays needed for training from the model
input_arrays = model.input_shapes
target_arrays = model.output_shapes


# %%
# Define the optimizer
optimizer = torch.optim.RAdam(model.parameters())

# Construct the datasplit
datasplit = CellMapDataSplit(
    input_arrays,
    target_arrays,
    classes,
    pad=True,
    csv_path="datasplit.csv",  # You need to provide this file (see cellmap_data.datasplit.make_datasplit_csv)
)

# Define the loss function
loss_function = BCELoss()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Move the model to the device and set it to training mode
model.to(device)
model.train()

# Construct the dataloader
train_loader = CellMapDataLoader(
    datasplit.train_datasets_combined,
    classes=classes,
    batch_size=batch_size,
    sampler=lambda: datasplit.train_datasets_combined.get_subset_random_sampler(
        steps * batch_size
    ),
    device=device,
)

# %%
# Training loop
losses = []  # List to store the losses
for epoch in range(epochs):
    # Redraw training examples
    train_loader.refresh()

    bar = tqdm(train_loader.loader, desc="Training")
    for batch in bar:
        targets = {k: batch[k] for k in model.output_keys}

        # Forward pass
        outputs = model(batch)

        # Compute the loss
        loss = loss_function(outputs, targets)

        # Backward pass
        loss.backward()

        # Store the loss
        losses.append(loss.item())

        # Update the weights
        optimizer.step()

        # Zero the gradients
        optimizer.zero_grad()

        # Update the progress bar
        bar.set_postfix(loss=loss.item())

# %%
# Plot the losses
import matplotlib.pyplot as plt

plt.plot(losses, xlabel="Step", ylabel="Loss")

# %%
