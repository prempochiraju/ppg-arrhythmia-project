# visualize_gradcam.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ppg_dataset import PPGDataset
from ppg_vgg_model import PPGVGGNet


class GradCAM1D:
    """
    Simple 1D Grad-CAM implementation for PPGVGGNet.
    """

    def __init__(self, model, target_layer_name: str = "features.0"):
        """
        target_layer_name is a dot path, e.g. "features.0" or "features.10".
        We'll hook that layer to read activations and gradients.
        """
        self.model = model
        self.model.eval()

        # Parse target layer from model via attribute/index path
        modules = target_layer_name.split(".")
        layer = model
        for m in modules:
            if m.isdigit():
                layer = layer[int(m)]
            else:
                layer = getattr(layer, m)
        self.target_layer = layer

        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; index 0 is the gradient w.r.t. the output
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x: torch.Tensor, class_idx: int = None):
        """
        x: (1, 1, 1000)
        class_idx: target class index; if None, use predicted class.
        Returns:
            cam: 1D numpy array of length L_feature (not 1000 yet!)
            class_idx: int
        """
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        loss = logits[0, class_idx]
        loss.backward()

        # gradients: (B, C, L), activations: (B, C, L)
        grads = self.gradients  # (1, C, L)
        acts = self.activations  # (1, C, L)

        # global-average-pool gradients over L dimension
        weights = torch.mean(grads, dim=2, keepdim=True)  # (1, C, 1)

        # weighted sum of activations
        cam = torch.sum(weights * acts, dim=1)  # (1, L)
        cam = F.relu(cam)
        cam = cam.squeeze(0)  # (L,)

        # normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "checkpoints", "ppg_vgg_best.pth")
    data_new_dir = os.path.join(base_dir, "data_new")
    test_path = os.path.join(data_new_dir, "test.npz")

    # For now, force CPU (your RTX 5060 + PyTorch combo is tricky)
    device = torch.device("cpu")
    print(f"Using device for Grad-CAM: {device}")

    # Load model
    model = PPGVGGNet(num_classes=6).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Load test dataset
    test_ds = PPGDataset(test_path)

    # pick an example index
    idx = 0  # change this manually to visualize different samples
    x, y_true = test_ds[idx]          # x: (1, 1000)
    x = x.unsqueeze(0).to(device)     # (1, 1, 1000)

    # Grad-CAM
    gradcam = GradCAM1D(model, target_layer_name="features.0")
    cam, class_idx = gradcam.generate(x)   # cam length = L_feature (e.g., 31)

    x_raw = test_ds.ppg[idx]               # normalized waveform (1000,)
    N = len(x_raw)

    # Upsample CAM from length L_feature -> 1000 using interpolation
    L_feat = len(cam)
    cam_x = np.linspace(0, N - 1, num=L_feat)
    t = np.arange(N) / 100.0   # assuming 100 Hz -> seconds
    cam_interp = np.interp(np.arange(N), cam_x, cam)

    print("\n=== Grad-CAM Diagnostics ===")
    print(f"True Label:        {y_true}")
    print(f"Predicted Label:   {class_idx}")
    print(f"CAM Raw Length:    {L_feat}")
    print(f"Signal Length:     {N}")
    print(f"CAM Value Range:   {cam.min():.4f} → {cam.max():.4f}")
    print("============================\n")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, x_raw, label="PPG (normalized)")
    plt.fill_between(t, 0, cam_interp, alpha=0.4, label="Grad-CAM importance")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / Importance")
    plt.title(f"PPG Segment with Grad-CAM (true={y_true}, pred={class_idx})")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(base_dir, f"gradcam_example_idx{idx}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Grad-CAM figure saved at:")
    print(f"  {out_path}\n")


if __name__ == "__main__":
    main()
