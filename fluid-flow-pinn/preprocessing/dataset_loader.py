"""PyTorch Dataset classes: FDSTDataset, UMNDataset, ShanghaiTechDataset."""

import os
import json
import glob
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

# ── Default ImageNet normalisation ────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _load_frame_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as uint8 RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _frame_to_tensor(frame: np.ndarray, transform: Callable) -> torch.Tensor:
    """Apply transform to an HxWx3 uint8 RGB numpy array → CHW float tensor."""
    return transform(frame)


def _upsample_tensor(
    t: torch.Tensor, height: int, width: int, mode: str = "bicubic"
) -> torch.Tensor:
    """Upsample a CHW tensor to (height, width)."""
    return F.interpolate(
        t.unsqueeze(0), size=(height, width), mode=mode, align_corners=False
    ).squeeze(0)


# ── Scene-aware sampler ───────────────────────────────────────────────────────

class SequentialSceneSampler(Sampler):
    """Yields indices that preserve temporal order within each scene.

    Scenes are shuffled between epochs (when shuffle=True), but frames
    within each scene always appear in order. This satisfies the constraint:
        "never shuffle frames independently — shuffle sequences, then
         iterate within each sequence in order."

    Usage:
        sampler = SequentialSceneSampler(dataset, shuffle=True)
        loader  = DataLoader(dataset, batch_size=4, sampler=sampler)

    On each new epoch call `sampler.set_epoch(epoch)` to get a different
    scene ordering (same API as DistributedSampler).
    """

    def __init__(self, dataset: "FDSTDataset", shuffle: bool = True) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        scene_names = list(self.dataset.scene_ranges.keys())
        if self.shuffle:
            rng = random.Random(self._epoch)
            rng.shuffle(scene_names)
        for scene in scene_names:
            start, end = self.dataset.scene_ranges[scene]
            yield from range(start, end)

    def __len__(self) -> int:
        return len(self.dataset)


# ── FDST Dataset ──────────────────────────────────────────────────────────────

class FDSTDataset(Dataset):
    """Consecutive frame pairs from the FDST dataset.

    Actual on-disk layout:
        <root>/
          train_data/
            1/   2/   ...   ← scene dirs (numbered)
              001.jpg  001.json
              002.jpg  002.json
              ...
          test_data/
            1/   2/   ...

    Each JSON file contains head-point annotations used to build a
    Gaussian-smoothed density map (sigma=15 px), downsampled to 1/8
    resolution to match CSRNet output.

    Each __getitem__ returns:
        {
            'frame_t':      FloatTensor (3, H, W)   normalised RGB,
            'frame_t1':     FloatTensor (3, H, W)   normalised RGB,
            'density_map':  FloatTensor (1, H//8, W//8),
            'scene':        str   (scene folder name),
            'idx':          int   (frame index within scene),
        }
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        density_sigma: float = 15.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform or _DEFAULT_TRANSFORM
        self.density_sigma = density_sigma
        self._samples: List[Dict] = []
        # Maps scene name → (start_idx, end_idx) in _samples (end exclusive)
        self.scene_ranges: Dict[str, Tuple[int, int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        # Support both "train_data" and "train" naming conventions
        split_dir = self.root / f"{self.split}_data"
        if not split_dir.exists():
            split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"FDST split directory not found. Tried:\n"
                f"  {self.root / f'{self.split}_data'}\n"
                f"  {self.root / self.split}"
            )

        for scene_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not scene_dir.is_dir():
                continue

            # Collect all .jpg files sorted by filename (001.jpg, 002.jpg …)
            frame_files = sorted(
                scene_dir.glob("*.jpg"),
                key=lambda p: p.stem,
            )
            if len(frame_files) < 2:
                continue

            scene_start = len(self._samples)
            for i in range(len(frame_files) - 1):
                ft_path = frame_files[i]
                ft1_path = frame_files[i + 1]
                json_path = ft_path.with_suffix(".json")
                self._samples.append(
                    {
                        "scene": scene_dir.name,
                        "frame_t": str(ft_path),
                        "frame_t1": str(ft1_path),
                        "json": str(json_path) if json_path.exists() else None,
                        "idx": i,
                    }
                )
            self.scene_ranges[scene_dir.name] = (scene_start, len(self._samples))

    def _build_density_from_json(
        self, json_path: Optional[str], h: int, w: int
    ) -> torch.Tensor:
        """Build a Gaussian-smoothed density map from a FDST VIA JSON file.

        FDST uses VGG Image Annotator (VIA) format — each entry is a dict
        keyed by "<filename><filesize>" containing bounding-box regions:
            {
              "001.jpg143318": {
                "regions": [
                  {"shape_attributes": {"name": "rect",
                                        "x": 373, "y": 18,
                                        "width": 38, "height": 36}, ...},
                  ...
                ]
              }
            }
        Head center = (x + width/2, y + height/2).
        """
        from scipy.ndimage import gaussian_filter

        cx_list: List[float] = []
        cy_list: List[float] = []

        if json_path is not None:
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # VIA format: outer dict has one key per image entry
                if isinstance(data, dict):
                    for entry in data.values():
                        if not isinstance(entry, dict):
                            continue
                        for region in entry.get("regions", []):
                            sa = region.get("shape_attributes", {})
                            if sa.get("name") == "rect":
                                cx = sa["x"] + sa["width"] / 2.0
                                cy = sa["y"] + sa["height"] / 2.0
                                cx_list.append(cx)
                                cy_list.append(cy)
                            elif sa.get("name") == "point":
                                cx_list.append(sa["cx"])
                                cy_list.append(sa["cy"])

            except (json.JSONDecodeError, OSError, KeyError):
                pass

        density = np.zeros((h, w), dtype=np.float32)
        for cx, cy in zip(cx_list, cy_list):
            x = int(round(cx))
            y = int(round(cy))
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            density[y, x] += 1.0

        density = gaussian_filter(density, sigma=self.density_sigma)

        # Downsample to 1/8 resolution to match CSRNet output
        dh, dw = max(1, h // 8), max(1, w // 8)
        density_t = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
        density_t = F.interpolate(
            density_t, size=(dh, dw), mode="bilinear", align_corners=False
        )
        return density_t.squeeze(0)  # (1, dh, dw)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self._samples[index]

        ft_np = _load_frame_rgb(sample["frame_t"])
        ft1_np = _load_frame_rgb(sample["frame_t1"])
        h, w = ft_np.shape[:2]

        frame_t = _frame_to_tensor(ft_np, self.transform)
        frame_t1 = _frame_to_tensor(ft1_np, self.transform)
        density_map = self._build_density_from_json(sample["json"], h, w)

        return {
            "frame_t": frame_t,
            "frame_t1": frame_t1,
            "density_map": density_map,
            "scene": sample["scene"],
            "idx": sample["idx"],
        }


# ── UMN Dataset ───────────────────────────────────────────────────────────────

class UMNDataset(Dataset):
    """Consecutive frame pairs from UMN scene clips (post-split).

    Expects either:
      (a) Pre-split scene .avi files:  umn/<scene_name>.avi
      (b) A single .avi at umn_root — will read all frames in order.

    Applies bicubic upsampling 320×240 → 640×480 (configurable) before
    returning tensors.

    Each __getitem__ returns:
        {
            'frame_t':     FloatTensor (3, 480, 640),
            'frame_t1':    FloatTensor (3, 480, 640),
            'density_map': FloatTensor (1, 1, 1)  — no GT density for UMN,
            'scene':       str,
            'idx':         int,
        }
    """

    def __init__(
        self,
        root: str | Path,
        target_height: int = 480,
        target_width: int = 640,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform or _DEFAULT_TRANSFORM
        self._samples: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        # Look for pre-split scene files
        scene_files = sorted(self.root.glob("scene*.avi"))
        if not scene_files:
            # Fall back to any .avi in root
            scene_files = sorted(self.root.glob("*.avi"))

        for vpath in scene_files:
            cap = cv2.VideoCapture(str(vpath))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for i in range(n - 1):
                self._samples.append(
                    {
                        "video": str(vpath),
                        "scene": vpath.stem,
                        "frame_t_idx": i,
                        "idx": i,
                    }
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self._samples[index]
        cap = cv2.VideoCapture(sample["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_t_idx"])
        ret0, f0 = cap.read()
        ret1, f1 = cap.read()
        cap.release()

        if not (ret0 and ret1):
            raise RuntimeError(
                f"Failed to read frames {sample['frame_t_idx']} "
                f"from {sample['video']}"
            )

        f0 = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)

        t0 = _frame_to_tensor(f0, self.transform)
        t1 = _frame_to_tensor(f1, self.transform)

        # Upsample 320×240 → 640×480 (bicubic)
        t0 = _upsample_tensor(t0, self.target_height, self.target_width)
        t1 = _upsample_tensor(t1, self.target_height, self.target_width)

        return {
            "frame_t": t0,
            "frame_t1": t1,
            "density_map": torch.zeros(1, 1, 1),
            "scene": sample["scene"],
            "idx": sample["idx"],
        }


# ── ShanghaiTech Dataset ──────────────────────────────────────────────────────

class ShanghaiTechDataset(Dataset):
    """Still-image density dataset from ShanghaiTech Part A or B.

    Expected layout (standard ShanghaiTech download):
        shanghaitech/
          part_A/
            train_data/
              images/    ← .jpg files
              ground_truth/  ← GT_<img_name>.mat files
            test_data/
              ...
          part_B/
            ...

    Density maps are constructed from dot annotations in the .mat files by
    spreading each head annotation over a small Gaussian kernel (sigma=15 px).

    This dataset has no temporal pairs — frame_t and frame_t1 are both the
    same image (evaluation-only dataset).

    Each __getitem__ returns:
        {
            'frame_t':     FloatTensor (3, H, W),
            'frame_t1':    FloatTensor (3, H, W),   ← same as frame_t
            'density_map': FloatTensor (1, H/8, W/8),
            'scene':       str  ('part_A' or 'part_B'),
            'idx':         int,
        }
    """

    def __init__(
        self,
        root: str | Path,
        part: str = "A",
        split: str = "test",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.part = f"part_{part}"
        self.split = f"{split}_data"
        self.transform = transform or _DEFAULT_TRANSFORM
        self._samples: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        img_dir = self.root / self.part / self.split / "images"
        gt_dir = self.root / self.part / self.split / "ground_truth"

        if not img_dir.exists():
            raise FileNotFoundError(f"ShanghaiTech images dir not found: {img_dir}")

        for img_path in sorted(img_dir.glob("*.jpg")):
            gt_path = gt_dir / f"GT_{img_path.name.replace('.jpg', '.mat')}"
            self._samples.append(
                {
                    "image": str(img_path),
                    "gt": str(gt_path) if gt_path.exists() else None,
                    "idx": len(self._samples),
                }
            )

    def _load_density_from_mat(
        self, gt_path: Optional[str], h: int, w: int
    ) -> torch.Tensor:
        if gt_path is None:
            dh, dw = max(1, h // 8), max(1, w // 8)
            return torch.zeros(1, dh, dw)

        mat = sio.loadmat(gt_path)
        # ShanghaiTech stores points under 'image_info' → 'location'
        try:
            points = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        except (KeyError, IndexError):
            points = np.empty((0, 2), dtype=np.float32)

        density = np.zeros((h, w), dtype=np.float32)
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            density[y, x] += 1.0

        # Smooth with a Gaussian kernel (sigma=15 px)
        from scipy.ndimage import gaussian_filter
        density = gaussian_filter(density, sigma=15)

        # Downsample to 1/8 resolution to match CSRNet output
        dh, dw = max(1, h // 8), max(1, w // 8)
        density_t = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
        density_t = F.interpolate(density_t, size=(dh, dw), mode="bilinear", align_corners=False)
        return density_t.squeeze(0)  # (1, dh, dw)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self._samples[index]
        img = _load_frame_rgb(sample["image"])
        h, w = img.shape[:2]

        frame_t = _frame_to_tensor(img, self.transform)
        density_map = self._load_density_from_mat(sample["gt"], h, w)

        return {
            "frame_t": frame_t,
            "frame_t1": frame_t,   # evaluation only — no temporal pair
            "density_map": density_map,
            "scene": self.part,
            "idx": sample["idx"],
        }
