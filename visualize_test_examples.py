import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ppg_dataset import PPGDataset
from ppg_vgg_model import PPGVGGNet


# Class index → name mapping (same as your report)
CLASS_NAMES = {
    0: "Normal",
    1: "PVC",
    2: "PAC",
    3: "VT",
    4: "SVT",
    5: "AFib",
}


def load_model_and_data(base_dir, device):
    ckpt_path = os.path.join(base_dir, "checkpoints", "ppg_vgg_best.pth")
    test_path = os.path.join(base_dir, "data_new", "test.npz")

    # Load dataset
    test_ds = PPGDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Load model
    model = PPGVGGNet(num_classes=6).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, test_ds, test_loader


def collect_predictions(model, test_loader, device):
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.append(preds)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def pick_examples(y_true, y_pred, num_correct_per_class=2, num_wrong=3):
    """
    Returns:
      - correct_examples: dict[class_idx] = [indices...]
      - wrong_examples: list of indices (misclassified)
    """
    correct_examples = {c: [] for c in range(6)}
    wrong_indices = []

    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            if len(correct_examples[t]) < num_correct_per_class:
                correct_examples[t].append(idx)
        else:
            wrong_indices.append(idx)

    # Optionally limit wrong examples
    if len(wrong_indices) > num_wrong:
        wrong_indices = wrong_indices[:num_wrong]

    return correct_examples, wrong_indices


def plot_example(ppg_signal, true_label, pred_label, idx, out_dir, fs=100.0):
    """
    ppg_signal: 1D numpy array (length 1000)
    true_label, pred_label: int
    """
    t = np.arange(len(ppg_signal)) / fs

    plt.figure(figsize=(10, 3))
    plt.plot(t, ppg_signal, linewidth=1.0)

    true_name = CLASS_NAMES[int(true_label)]
    pred_name = CLASS_NAMES[int(pred_label)]

    title = f"Test Sample idx={idx} | True: {true_name} ({true_label}) | Pred: {pred_name} ({pred_label})"
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized PPG")

    plt.tight_layout()

    fname = f"test_example_idx{idx}_true{true_label}_pred{pred_label}.png"
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, test_ds, test_loader = load_model_and_data(base_dir, device)
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # Pick some examples
    correct_per_class, wrong_examples = pick_examples(
        y_true, y_pred,
        num_correct_per_class=2,  # change if you want more per class
        num_wrong=4               # change if you want more misclassified
    )

    out_dir = os.path.join(base_dir, "test_examples")
    os.makedirs(out_dir, exist_ok=True)

    # Plot correctly classified examples for each class
    print("\n=== Correctly classified examples per class ===")
    for cls_idx in range(6):
        indices = correct_per_class[cls_idx]
        if not indices:
            print(f"No correct examples found for class {cls_idx} ({CLASS_NAMES[cls_idx]})")
            continue

        print(f"Class {cls_idx} ({CLASS_NAMES[cls_idx]}): indices {indices}")
        for idx in indices:
            # test_ds.ppg is the raw normalized waveform, shape (N, 1000)
            ppg_signal = test_ds.ppg[idx]  # numpy array
            plot_example(ppg_signal, y_true[idx], y_pred[idx], idx, out_dir)

    # Plot some misclassified examples
    print("\n=== Misclassified examples ===")
    for idx in wrong_examples:
        ppg_signal = test_ds.ppg[idx]
        plot_example(ppg_signal, y_true[idx], y_pred[idx], idx, out_dir)


if __name__ == "__main__":
    main()
