"""
Dataset loading for the deepfake detector.

Expects a directory layout of:

    <split>/
        real/   *.jpg|*.png|...
        fake/   *.jpg|*.png|...

This is the standard layout used by most public deepfake datasets
(FaceForensics++, Celeb-DF, DFDC-derived crops, etc.) once you've
extracted frames, so pointing DATASET_DIR at a real dataset in
config.py should work with little to no change.
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DeepfakeDataset(Dataset):
    """
    Loads (image, label) pairs from a `real/` and `fake/` subfolder.
    label = 0 for real, 1 for fake.
    """

    def __init__(self, root_dir: str, image_size: int = 224, train: bool = True):
        self.samples = []
        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMG_EXTS):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {root_dir}. Expected 'real/' and 'fake/' "
                f"subfolders containing image files."
            )

        self.transform = build_transforms(image_size, train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label

    def class_balance(self):
        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        return n_real, n_fake
