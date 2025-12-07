# train_ppg_vgg.py  (with safe GPU fallback + training curves)

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt   # <-- NEW

from ppg_dataset import build_dataloaders
from ppg_vgg_model import PPGVGGNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, running_correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)

            running_correct += (preds == y).sum().item()
            total += y.size(0)

            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return (
        running_loss / total,
        running_correct / total,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


def main():
    # -----------------------------
    # Device selection + safe fallback
    # -----------------------------
    print("Checking CUDA availability...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA is available.")
        print(f"Initial device: {device}")
        print(f"GPU Name: {gpu_name}")
        print(f"Total GPU Memory: {total_mem:.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    # -----------------------------
    # Paths + Hyperparameters
    # -----------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    num_classes = 6

    batch_size = 256 if device.type == "cuda" else 128
    num_epochs = 50
    lr = 1e-3
    weight_decay = 1e-4

    print(f"\nBatch size (initial): {batch_size}")

    # -----------------------------
    # Data
    # -----------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        num_workers=4 if device.type == "cuda" else 0,
    )

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = PPGVGGNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # -----------------------------
    # Try a dummy forward to verify GPU works
    # -----------------------------
    if device.type == "cuda":
        print("\nTesting a dummy forward on GPU to check compatibility...")
        try:
            model.eval()
            dummy_x, _ = next(iter(train_loader))
            dummy_x = dummy_x.to(device)
            with torch.no_grad():
                _ = model(dummy_x)
            print("Dummy forward on GPU succeeded. Training will run on CUDA.\n")
        except Exception as e:
            print("\n!!! GPU forward failed, falling back to CPU !!!")
            print("Error was:\n", repr(e))
            device = torch.device("cpu")
            model = model.to(device)
            print(f"\nUsing device: {device}\n")
    else:
        print("\nUsing CPU from the start.\n")

    # -----------------------------
    # Training Loop (with history)
    # -----------------------------
    best_val_acc = 0.0
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ppg_vgg_best.pth")

    # history lists
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting training...\n")

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = eval_one_epoch(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - start
        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train: {train_loss:.4f}/{train_acc:.4f} | "
            f"Val: {val_loss:.4f}/{val_acc:.4f} | "
            f"{epoch_time:.1f}s"
        )

        # store in history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ==> Saved new best model: {ckpt_path}")

    print(f"\nTraining finished. Best Val Accuracy = {best_val_acc:.4f}")

    # -----------------------------
    # Save training curves
    # -----------------------------
    epochs = np.arange(1, num_epochs + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    loss_fig_path = os.path.join(base_dir, "training_loss_curve.png")
    plt.savefig(loss_fig_path, dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_fig_path = os.path.join(base_dir, "training_accuracy_curve.png")
    plt.savefig(acc_fig_path, dpi=200)
    plt.close()

    print("\nSaved training curves:")
    print(f"  Loss curve:      {loss_fig_path}")
    print(f"  Accuracy curve:  {acc_fig_path}")

    # -----------------------------
    # Test Evaluation
    # -----------------------------
    print("\nLoading best model for test evaluation...\n")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_loss, test_acc, y_true, y_pred = eval_one_epoch(
        model, test_loader, criterion, device
    )

    print("\nTEST RESULTS:")
    print(f"Test Loss = {test_loss:.4f}")
    print(f"Test Acc  = {test_acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
