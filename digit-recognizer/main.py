# %% [markdown]
"""
# Digit Recognizer
## Simple CNN with an MLP Classifier

- Public LB Score: 0.98910

"""

# %%
from typing import List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"

# %%
params = {
    "num_epochs": 40,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "activation": "ReLU",
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# %%
SEED = 503
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Digit Recognizer")
MLFLOW_RUN_NAME = "DR4"
MLFLOW_RUN_DESCRIPTION = ""

# %% [markdown]
"""
## PyTorch Dataset and DataLoader
"""


# %%
def columns_to_image(columns: List) -> np.ndarray:
    """Transform the list of 784 items into a 2D numpy array."""
    image = np.array(columns)
    image = image.reshape((28, 28))
    return image


def normalize(
    image: torch.Tensor,
    mean: float = 0.5,
    std: float = 0.5,
) -> torch.Tensor:
    return F.normalize(image, mean=[mean], std=[std])


class DigitDataset(data.Dataset):
    def __init__(
        self,
        images: List,
        labels: List,
        train: bool = True,
    ) -> None:
        super(DigitDataset, self).__init__()
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.train = train
        self.augment = transforms.Compose(
            [
                transforms.RandomCrop(size=28, padding=1),
            ]
        )

    def __getitem__(self, index: int) -> Tuple:
        image = columns_to_image(self.images[index].tolist())
        image = torch.from_numpy(image).float().unsqueeze(0)

        if self.train:
            image = self.augment(image)

        image = normalize(image)

        label = self.labels[index]
        return (image, label)

    def __len__(self) -> int:
        return len(self.images)


class DigitDataLoader(data.DataLoader):
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = params["batch_size"],
        *args,
        **kwargs,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        super(DigitDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    def __len__(self) -> int:
        assert self.batch_size is not None
        return int(len(self.dataset) / self.batch_size)  # type:ignore


# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_X = train_df.loc[:, train_df.columns != "label"].to_numpy()
train_y = train_df["label"].to_numpy()

train_X, val_X, train_y, val_y = train_test_split(
    train_X,
    train_y,
    test_size=0.2,
    shuffle=True,
)


train_dataset = DigitDataset(train_X, train_y)
val_dataset = DigitDataset(val_X, val_y, train=False)

train_dataloader = DigitDataLoader(train_dataset, shuffle=True)
val_dataloader = DigitDataLoader(val_dataset)


# %% [markdown]
"""
## PyTorch Model
"""


# %%
class DigitClassifier(nn.Module):
    def __init__(self, activation: str, *args, **kwargs) -> None:
        super(DigitClassifier, self).__init__(*args, **kwargs)

        match activation:
            case "ReLU":
                self.activation = nn.ReLU
            case "LeakyReLU":
                self.activation = nn.LeakyReLU
            case _:
                print(
                    f"No activation named {activation}.",
                    "Using default ReLU function.",
                )
                self.activation = nn.ReLU

        self.convolutional_layers = nn.Sequential(
            self.build_convolution_block(1, 30),
            self.build_convolution_block(30, 30),
            self.build_convolution_block(30, 15),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=135, out_features=75),
            nn.BatchNorm1d(num_features=75),
            self.activation(),
            nn.Linear(in_features=75, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layers(x)
        return x

    def build_convolution_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            self.activation(),
            nn.MaxPool2d(kernel_size=2),
        )


# %% [markdown]
"""
## Metrics
"""


# %%
def accuracy(pred: torch.Tensor, real: torch.Tensor) -> Tuple[int, int]:
    """Get the accuracy of two tensors.

    Returns:
        Tuple: (number of right elements, number of wrong elements)

    """
    total_items = pred.shape[0]
    total_right = int((torch.argmax(pred, dim=1) == real).sum().item())
    return (total_right, total_items - total_right)


# %% [markdown]
"""
## Training Procedure
"""


# %%
def train_one_epoch(epoch: int) -> float:
    model.train()
    total_iter, total_loss, total_right, total_wrong = 0, 0.0, 0, 0
    for _, (image, label) in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
    ):
        image, label = image.to(DEVICE), label.to(DEVICE)

        prediction = model(image)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_iter += 1
        total_loss += loss.item()

        right, wrong = accuracy(prediction, label)
        total_right += right
        total_wrong += wrong

    train_loss = total_loss / total_iter
    train_accuracy = total_right / (total_right + total_wrong)

    print(f"[TRAIN] Epoch {epoch}")
    print(f"\tLoss: {train_loss:4f}")
    mlflow.log_metric("train_loss", train_loss, step=epoch)

    print(f"\tAcc : {train_accuracy:4f}%")
    mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

    model.eval()
    total_iter, total_loss, total_right, total_wrong = 0, 0.0, 0, 0
    with torch.no_grad():
        for _, (image, label) in tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader),
        ):
            image, label = image.to(DEVICE), label.to(DEVICE)

            prediction = model(image)
            loss = criterion(prediction, label)

            total_iter += 1
            total_loss += loss.item()

            right, wrong = accuracy(prediction, label)
            total_right += right
            total_wrong += wrong

    val_loss = total_loss / total_iter
    val_accuracy = total_right / (total_right + total_wrong)

    print(f"[VAL] Epoch {epoch}")
    print(f"\tLoss: {val_loss:4f}")
    mlflow.log_metric("val_loss", val_loss, step=epoch)

    print(f"\tAcc : {val_accuracy:4f}%")
    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

    return val_loss


# %%
model = DigitClassifier(activation=params["activation"]).to(DEVICE)
optimizer = torch.optim.AdamW(
    lr=params["learning_rate"],
    params=model.parameters(),
)
criterion = nn.CrossEntropyLoss().to(DEVICE)

best_validation_loss = 1000000

with mlflow.start_run(
    run_name=MLFLOW_RUN_NAME,
    description=MLFLOW_RUN_DESCRIPTION,
):
    for key, value in params.items():
        mlflow.log_param(key, value)
    for i in range(params["num_epochs"]):
        val_loss = train_one_epoch(i)

        if val_loss >= best_validation_loss:
            print("Validation loss did not decrease. Early stopping.")
            break
        else:
            best_validation_loss = val_loss


# %% [markdown]
"""
## Generate Submission
"""


# %%
def generate_submission(
    model: nn.Module,
    test_X: pd.DataFrame,
    submission_file="./submission.csv",
):
    print("Generating submission file ...")
    model.eval()

    results = []

    for _, row in tqdm(test_X.iterrows(), total=len(test_X)):
        image = (
            torch.from_numpy(columns_to_image(row.tolist()))
            .float()
            .to(DEVICE)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
        )

        image = normalize(image)

        label = model(image).squeeze()
        results.append(torch.argmax(label).item())

    print(f"Saving submission file to {submission_file} ...")
    with open(submission_file, "w") as f:
        contents = "ImageId,Label\n" + "\n".join(
            [f"{index + 1},{item}" for index, item in enumerate(results)],
        )
        f.write(contents)


generate_submission(model, test_df)
