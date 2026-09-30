"""
HW3 Question 2: Transfer Learning - Freeze vs Fine-Tune

Dataset:
    sklearn Digits, restricted to two classes (0 and 1).
Model:
    Pretrained ResNet18.
Experiments:
    A. Freeze the convolutional/base layers; train only the final classifier.
    B. Unfreeze layer4 and the classifier; train both.

The script prints trainable parameter counts, training time, and validation
accuracy, and saves a training-loss plot plus a CSV summary.
"""

import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Dataset ----------------
digits = load_digits()
mask = np.isin(digits.target, [0, 1])
X = digits.images[mask].astype(np.float32) / 16.0
y = digits.target[mask].astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# Convert 8x8 grayscale images to 224x224 3-channel tensors.
# This lets the images be processed by standard ResNet18.
def make_tensor(images):
    t = torch.tensor(images).unsqueeze(1)       # N,1,8,8
    t = t.repeat(1, 3, 1, 1)                    # N,3,8,8
    t = torch.nn.functional.interpolate(
        t, size=(224, 224), mode="bilinear", align_corners=False
    )

    # ImageNet normalization expected by pretrained ResNet.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std

train_ds = TensorDataset(make_tensor(X_train), torch.tensor(y_train))
val_ds = TensorDataset(make_tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- Model ----------------
def build_model(experiment):
    # Download/use standard pretrained ImageNet weights.
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers first.
    for p in model.parameters():
        p.requires_grad = False

    # Replace the classifier for two classes.
    model.fc = nn.Linear(model.fc.in_features, 2)

    if experiment == "fine_tune":
        # Unfreeze the last convolutional block.
        for p in model.layer4.parameters():
            p.requires_grad = True

    return model.to(device)

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    losses = []
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            samples += labels.size(0)

        epoch_loss = running_loss / samples
        losses.append(epoch_loss)

        accuracy = evaluate(model)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Validation Accuracy: {accuracy:.2f}%"
        )

    elapsed = time.perf_counter() - start
    final_accuracy = evaluate(model)
    return losses, elapsed, final_accuracy

# ---------------- Run experiments ----------------
results = []
all_losses = {}

for experiment, label in [
    ("feature_extraction", "Frozen Feature Extractor"),
    ("fine_tune", "Fine-Tuned Network")
]:
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    model = build_model(experiment)
    trainable = count_trainable(model)

    losses, elapsed, accuracy = train_model(model)

    results.append({
        "Method": label,
        "Trainable Parameters": trainable,
        "Training Time (seconds)": round(elapsed, 2),
        "Validation Accuracy (%)": round(accuracy, 2)
    })
    all_losses[label] = losses

# ---------------- Results table ----------------
results_df = pd.DataFrame(results)
print("\nFinal comparison:")
print(results_df.to_string(index=False))
results_df.to_csv("transfer_learning_results.csv", index=False)

# ---------------- Loss plot ----------------
plt.figure(figsize=(8, 5))
for label, losses in all_losses.items():
    plt.plot(range(1, EPOCHS + 1), losses, marker="o", label=label)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss: Frozen vs Fine-Tuned ResNet18")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=200)
plt.show()
