import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os
import sys

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.pth"
LABELS_FILE = "gesture_labels.json"

# --- hyperparameters ---
HIDDEN_1 = 128
HIDDEN_2 = 64
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEST_SPLIT = 0.2


class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run collect_gestures.py first.")
        sys.exit(1)

    labels = []
    features = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            labels.append(row[0])
            features.append([float(v) for v in row[1:]])

    print(f"Loaded {len(labels)} samples from {path}")
    return np.array(features, dtype=np.float32), labels


def main():
    X, raw_labels = load_data(DATA_FILE)
    input_size = X.shape[1]

    # Encode string labels -> integers
    le = LabelEncoder()
    y = le.fit_transform(raw_labels)
    num_classes = len(le.classes_)

    print(f"Classes ({num_classes}): {list(le.classes_)}")
    for cls in le.classes_:
        print(f"  {cls}: {sum(1 for l in raw_labels if l == cls)} samples")

    if num_classes < 2:
        print("Error: need at least 2 gesture classes to train.")
        sys.exit(1)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = GestureNet(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  train_acc={train_acc:.3f}")

    # --- evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

    test_acc = correct / total
    print(f"\nTest accuracy: {test_acc:.3f}  ({correct}/{total})")

    # --- per-class accuracy ---
    class_correct = {c: 0 for c in le.classes_}
    class_total = {c: 0 for c in le.classes_}
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            for pred, true in zip(preds, yb):
                label = le.classes_[true.item()]
                class_total[label] += 1
                if pred == true:
                    class_correct[label] += 1

    print("\nPer-class accuracy:")
    for cls in le.classes_:
        n = class_total[cls]
        acc = class_correct[cls] / n if n > 0 else 0
        print(f"  {cls}: {acc:.3f}  ({class_correct[cls]}/{n})")

    # --- save model + labels ---
    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "num_classes": num_classes,
    }, MODEL_FILE)

    with open(LABELS_FILE, "w") as f:
        json.dump(list(le.classes_), f)

    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Labels saved to {LABELS_FILE}")


if __name__ == "__main__":
    main()
