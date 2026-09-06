"""
Grad-CAM: produces a heatmap over the input image showing which regions
most influenced the model's prediction. For a deepfake detector this is
not a nice-to-have -- a bare "87% fake" score is not actionable or
trustworthy on its own. Showing *where* the model saw suspicious
texture (e.g. around the eyes, mouth, or a blending seam) lets a human
sanity-check the verdict.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        """
        input_tensor: (1, C, H, W), requires no grad set beforehand.
        Returns a (H, W) numpy heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)
        score = output[:, class_idx].sum()
        score.backward()

        gradients = self.gradients[0]        # (C, h, w)
        activations = self.activations[0]     # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def overlay_heatmap(pil_image, cam: np.ndarray, alpha: float = 0.45):
    """Resize a Grad-CAM heatmap to the image size and blend it on top."""
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlaid = (img * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return overlaid
