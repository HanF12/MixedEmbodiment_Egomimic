#!/usr/bin/env python
"""
t-SNE of ResNet backbone embeddings for Combined_relative_3cam / MixedEmbodiment.

For each sampled frame, runs the **bird-view** image through the trained ResNet
backbone (never wrist cams), applies global average pooling, and collects one
embedding per frame. Embeddings are stacked and reduced with PCA, t-SNE, and UMAP
(all n_components=2), then scatter-plotted colored by dataset type.

Scatter plots are colored by dataset type:
  - Combined: human vs robot (teleop)
  - MixedEmbodiment: teleop, human, left robot + right hand, right robot + left hand

Backbone extractors are pluggable via the BackboneFeatureExtractor protocol so
other encoders can be swapped in without changing the sampling / t-SNE pipeline.

Example:
  python recording/data_analysis/plot_resnet_tsne.py \\
    --checkpoints \\
      Combined_relative_3cam/combined_act_epoch_8000.pth \\
      MixedEmbodiment/mixed_act_epoch_6000_24.pth \\
    --frame_stride 20 --max_demos 2
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MixedEmbodiment.build_sync import (  # noqa: E402
    _infer_preset,
    _resolve_eef_dir,
    _resolve_pose_dir,
    build_mixed_sync_csvs,
)
from MixedEmbodiment.config import (  # noqa: E402
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_HUMAN,
    EMBODIMENT_MIXED,
    EMBODIMENT_ROBOT,
    MODEL_CAMERA_NAMES,
    camera_mask_tensor,
    stack_camera_tensors,
)
from MixedEmbodiment.dataloader_human import HumanEpisodeDataset  # noqa: E402
from MixedEmbodiment.dataloader_mixed import MixedEpisodeDataset  # noqa: E402
from MixedEmbodiment.dataloader_robot import RobotEpisodeDataset  # noqa: E402
from MixedEmbodiment.dataloader_utils import (  # noqa: E402
    demo_id_from_hash_filename,
    demo_id_from_robot_eef_npz,
    load_frame,
    zero_rgb_like,
)
from MixedEmbodiment.training_combined import (  # noqa: E402
    build_human_sync_csvs,
    build_robot_sync_csvs,
    resolve_human_pose_dir,
    resolve_robot_eef_dir,
)

_TS_RE = re.compile(r"(\d{14})$")


def _timestamp_suffix(demo_id: str) -> str | None:
    m = _TS_RE.search(demo_id)
    return m.group(1) if m else None


def remap_eef_dir_to_session_ids(
    mixed_root: Path,
    eef_dir: Path,
    staging_root: Path,
) -> Path:
    """
    Mixed sessions sometimes store EEF NPZs under teleop_bimanual_<ts>_... while
    videos / sync IDs use <session_kind>_<ts>. Symlink NPZs into a staging dir
    renamed so demo_id_from_robot_eef_npz matches the bird demo IDs.
    """
    bird_npy = mixed_root / "bird-realsense-data" / "npy"
    bird_ids = [demo_id_from_hash_filename(p) for p in sorted(bird_npy.glob("*.npy"))]
    eef_by_id = {demo_id_from_robot_eef_npz(p): p for p in sorted(eef_dir.glob("*.npz"))}

    # Already aligned — no remapping needed.
    if set(bird_ids) & set(eef_by_id):
        return eef_dir

    eef_by_ts = {_timestamp_suffix(k): p for k, p in eef_by_id.items() if _timestamp_suffix(k)}
    staging = staging_root / f"{mixed_root.parent.name}_{mixed_root.name}_eef_remapped"
    if staging.exists():
        for p in staging.glob("*.npz"):
            p.unlink()
    else:
        staging.mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for bird_id in bird_ids:
        ts = _timestamp_suffix(bird_id)
        if ts is None or ts not in eef_by_ts:
            continue
        src = eef_by_ts[ts]
        # Preserve a suffix that demo_id_from_robot_eef_npz strips correctly.
        dst = staging / f"{bird_id}_arm_fk_pose_targetframe_commonframe.npz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        n_linked += 1

    if n_linked == 0:
        raise FileNotFoundError(
            f"Could not remap EEF NPZs under {eef_dir} to bird demo IDs in {mixed_root}"
        )
    print(f"Remapped {n_linked} EEF NPZs -> {staging} (timestamp match; original prefix differed)")
    return staging

# ---------------------------------------------------------------------------
# Dataset labels / colors
# ---------------------------------------------------------------------------

DATASET_TYPES_COMBINED = ("teleop", "human")
DATASET_TYPES_MIXED = (
    "teleop",
    "human",
    "left_robot_right_hand",
    "right_robot_left_hand",
)
DATASET_COLORS = {
    "teleop": "#1f77b4",
    "human": "#2ca02c",
    "left_robot_right_hand": "#ff7f0e",
    "right_robot_left_hand": "#d62728",
}
DATASET_LABELS = {
    "teleop": "robot (teleop)",
    "human": "human",
    "left_robot_right_hand": "left robot + right hand",
    "right_robot_left_hand": "right robot + left hand",
}


@dataclass
class FrameRef:
    """Lightweight pointer into an episode dataset (no image tensors)."""

    dataset_type: str
    embodiment: int
    demo_idx: int
    start_in_ep: int
    progress: float
    # Which loader + builder to use; resolved at extract time.
    source: str  # teleop | human | left_robot_right_hand | right_robot_left_hand


class BuildArgs:
    """Minimal args object matching ACT build() expectations."""

    def __init__(self, num_queries: int, backbone: str = "resnet18"):
        self.num_queries = int(num_queries)
        self.camera_names = list(MODEL_CAMERA_NAMES)
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        self.position_embedding = "sine"
        self.backbone = backbone
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False


# ---------------------------------------------------------------------------
# Pluggable backbone extractors
# ---------------------------------------------------------------------------


class BackboneFeatureExtractor(ABC):
    """Interface for turning images into a single embedding vector per frame."""

    name: str = "base"

    @abstractmethod
    @torch.no_grad()
    def extract(self, images: torch.Tensor, camera_mask: torch.Tensor | None = None) -> np.ndarray:
        """
        Args:
            images: [num_cams, 3, H, W] or [1, num_cams, 3, H, W]
            camera_mask: optional [num_cams] or [1, num_cams] (1=active)
        Returns:
            float32 vector [D]
        """


BIRD_CAM_ID = 0  # MODEL_CAMERA_NAMES[0] == "bird"; wrists are never used for embeddings


class ResNetGAPExtractor(BackboneFeatureExtractor):
    """
    Global-average-pool the final ResNet feature map from the bird-view backbone
    only (model.backbones[0]). Wrist cameras are ignored.
    """

    name = "resnet_gap"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def extract(self, images: torch.Tensor, camera_mask: torch.Tensor | None = None) -> np.ndarray:
        del camera_mask  # bird-only; mask unused
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=False)
        if images.shape[0] != 1:
            raise ValueError(f"ResNetGAPExtractor expects batch size 1, got {images.shape[0]}")

        features, _ = self.model.backbones[BIRD_CAM_ID](images[:, BIRD_CAM_ID])
        gap = features[0].mean(dim=(2, 3))[0]  # [C]
        out = gap.detach().float().cpu().numpy().astype(np.float32)
        del images, features, gap
        return out


class InputProjGAPExtractor(BackboneFeatureExtractor):
    """Bird-view backbone + input_proj, then GAP. Wrist cameras ignored."""

    name = "input_proj_gap"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def extract(self, images: torch.Tensor, camera_mask: torch.Tensor | None = None) -> np.ndarray:
        del camera_mask
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=False)
        if images.shape[0] != 1:
            raise ValueError(f"InputProjGAPExtractor expects batch size 1, got {images.shape[0]}")

        features, _ = self.model.backbones[BIRD_CAM_ID](images[:, BIRD_CAM_ID])
        proj = self.model.input_proj(features[0])
        gap = proj.mean(dim=(2, 3))[0]
        out = gap.detach().float().cpu().numpy().astype(np.float32)
        del images, features, proj, gap
        return out


EXTRACTOR_REGISTRY: dict[str, type[BackboneFeatureExtractor]] = {
    "resnet_gap": ResNetGAPExtractor,
    "input_proj_gap": InputProjGAPExtractor,
}


def build_extractor(
    kind: str,
    model: torch.nn.Module,
    *,
    device: torch.device | None = None,
) -> BackboneFeatureExtractor:
    if kind not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor '{kind}'. Choose from {sorted(EXTRACTOR_REGISTRY)}")
    return EXTRACTOR_REGISTRY[kind](model, device=device)


# ---------------------------------------------------------------------------
# Path / model helpers
# ---------------------------------------------------------------------------


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def infer_model_pkg(checkpoint: Path) -> str:
    parts = checkpoint.resolve().parts
    for name in (
        "MixedEmbodiment_dual",
        "MixedEmbodiment",
        "Combined_relative_3cam",
        "Combined_relative_xyz_3cam",
        "Combined_relative",
    ):
        if name in parts:
            return name
    raise ValueError(f"Cannot infer model package from checkpoint path: {checkpoint}")


def load_build_fn(model_pkg: str):
    mod = importlib.import_module(f"{model_pkg}.core")
    return mod.build


def dataset_types_for_pkg(model_pkg: str) -> tuple[str, ...]:
    if model_pkg.startswith("MixedEmbodiment"):
        return DATASET_TYPES_MIXED
    return DATASET_TYPES_COMBINED


# ---------------------------------------------------------------------------
# Lightweight frame refs + on-demand image decode
# ---------------------------------------------------------------------------


def _progress(start: int, ep_len: int) -> float:
    return float(start) / float(max(1, ep_len - 1)) if ep_len > 1 else 0.0


def collect_strided_refs(
    *,
    robot_ds: RobotEpisodeDataset,
    human_ds: HumanEpisodeDataset,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
    frame_stride: int,
    max_frames_per_type: int | None,
    include_mixed: bool,
) -> list[FrameRef]:
    """Index every Nth frame — metadata only, no image tensors."""
    stride = max(1, int(frame_stride))

    def from_episode_ds(ds, dataset_type: str, embodiment: int) -> list[FrameRef]:
        out: list[FrameRef] = []
        for ep in range(int(ds.num_demos)):
            ep_len = int(ds.demo_lengths[ep])
            for start in range(0, ep_len, stride):
                out.append(
                    FrameRef(
                        dataset_type=dataset_type,
                        embodiment=embodiment,
                        demo_idx=ep,
                        start_in_ep=start,
                        progress=_progress(start, ep_len),
                        source=dataset_type,
                    )
                )
                if max_frames_per_type is not None and len(out) >= max_frames_per_type:
                    return out
        return out

    refs: list[FrameRef] = []
    refs += from_episode_ds(robot_ds, "teleop", EMBODIMENT_ROBOT)
    refs += from_episode_ds(human_ds, "human", EMBODIMENT_HUMAN)
    if include_mixed:
        if mixed_lr is None or mixed_rl is None:
            raise ValueError("Mixed datasets required when include_mixed=True")
        refs += from_episode_ds(mixed_lr, "left_robot_right_hand", EMBODIMENT_MIXED)
        refs += from_episode_ds(mixed_rl, "right_robot_left_hand", EMBODIMENT_MIXED)
    return refs


def load_bird_image_for_ref(
    ref: FrameRef,
    *,
    robot_ds: RobotEpisodeDataset,
    human_ds: HumanEpisodeDataset,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode + normalize the bird-view frame only.
    Wrist slots are always zeroed — never read from wrist videos.
    Returns (images [3,3,H,W], camera_mask [3] bird-only).
    """
    if ref.source == "teleop":
        ds = robot_ds
    elif ref.source == "human":
        ds = human_ds
    elif ref.source == "left_robot_right_hand":
        ds = mixed_lr
    elif ref.source == "right_robot_left_hand":
        ds = mixed_rl
    else:
        raise ValueError(f"Unknown source {ref.source}")
    assert ds is not None

    sample_idx = ds.demo_start_idx[ref.demo_idx] + ref.start_in_ep
    bird_np = load_frame(ds.bird_frames[sample_idx])
    zero_np = zero_rgb_like(bird_np)
    images = stack_camera_tensors(
        ds.image_transform(bird_np),
        ds.image_transform(zero_np),
        ds.image_transform(zero_np),
    )
    # Bird-only mask regardless of embodiment (wrists never fed to the backbone).
    return images, camera_mask_tensor(EMBODIMENT_HUMAN)


@torch.no_grad()
def extract_embeddings_streaming(
    refs: list[FrameRef],
    extractor: BackboneFeatureExtractor,
    *,
    robot_ds: RobotEpisodeDataset,
    human_ds: HumanEpisodeDataset,
    mixed_lr: MixedEpisodeDataset | None,
    mixed_rl: MixedEpisodeDataset | None,
    progress_every: int = 25,
) -> np.ndarray:
    """Decode one bird frame at a time, extract, discard tensors. Keeps only [N,D] floats."""
    embeddings: list[np.ndarray] = []
    n = len(refs)
    for i, ref in enumerate(refs):
        images, cam_mask = load_bird_image_for_ref(
            ref,
            robot_ds=robot_ds,
            human_ds=human_ds,
            mixed_lr=mixed_lr,
            mixed_rl=mixed_rl,
        )
        emb = extractor.extract(images, cam_mask)
        embeddings.append(emb)
        del images
        if (i + 1) % progress_every == 0 or i == 0 or i + 1 == n:
            print(f"  extracted {i + 1}/{n} (bird only)", flush=True)
        if (i + 1) % 100 == 0:
            gc.collect()
    return np.stack(embeddings, axis=0)


# ---------------------------------------------------------------------------
# 2D reducers + plotting
# ---------------------------------------------------------------------------


def run_tsne(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    random_state: int = 0,
    n_components: int = 2,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be [N,D], got {x.shape}")
    n = x.shape[0]
    perp = min(float(perplexity), max(5.0, (n - 1) / 3.0))
    if n < 4:
        raise RuntimeError(f"Need at least 4 samples for t-SNE, got {n}")
    print(f"  TSNE: N={n} D={x.shape[1]} perplexity={perp} random_state={random_state}", flush=True)
    reducer = TSNE(
        n_components=int(n_components),
        perplexity=perp,
        random_state=int(random_state),
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(x).astype(np.float32)


def run_pca(
    features: np.ndarray,
    *,
    random_state: int = 0,
    n_components: int = 2,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be [N,D], got {x.shape}")
    n = x.shape[0]
    n_comp = min(int(n_components), n, x.shape[1])
    if n_comp < 2:
        raise RuntimeError(f"Need at least 2 samples/dims for PCA, got N={n} D={x.shape[1]}")
    print(f"  PCA: N={n} D={x.shape[1]} n_components={n_comp} random_state={random_state}", flush=True)
    reducer = PCA(n_components=n_comp, random_state=int(random_state))
    return reducer.fit_transform(x).astype(np.float32)


def run_umap(
    features: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
    n_components: int = 2,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"features must be [N,D], got {x.shape}")
    n = x.shape[0]
    if n < 4:
        raise RuntimeError(f"Need at least 4 samples for UMAP, got {n}")
    # n_neighbors must be < N
    neighbors = max(2, min(int(n_neighbors), n - 1))
    print(
        f"  UMAP: N={n} D={x.shape[1]} n_neighbors={neighbors} min_dist={min_dist} "
        f"random_state={random_state}",
        flush=True,
    )
    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    return reducer.fit_transform(x).astype(np.float32)


def scatter_by_dataset(
    xy: np.ndarray,
    types: Sequence[str],
    *,
    dataset_order: Sequence[str],
    title: str,
    out_path: Path,
    xlabel: str = "dim 1",
    ylabel: str = "dim 2",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    for dt in dataset_order:
        mask = np.asarray([t == dt for t in types], dtype=bool)
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=22,
            c=DATASET_COLORS[dt],
            label=DATASET_LABELS[dt],
            alpha=0.8,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="t-SNE of ResNet GAP backbone embeddings across embodiment datasets"
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "Combined_relative_3cam/combined_act_epoch_8000.pth",
            "MixedEmbodiment/mixed_act_epoch_6000_24.pth",
        ],
    )
    p.add_argument(
        "--model_pkgs",
        nargs="*",
        default=None,
        help="Optional override, parallel to --checkpoints",
    )
    p.add_argument("--teleop_root", type=str, default="recording/sessions/teleop_bimanual/0714")
    p.add_argument("--human_root", type=str, default="recording/sessions/human_hands_bimanual_raw/0714")
    p.add_argument(
        "--mixed_lr_root",
        type=str,
        default="recording/sessions/left_robot_right_hand/0729",
        help="left robot + right human",
    )
    p.add_argument(
        "--mixed_rl_root",
        type=str,
        default="recording/sessions/right_robot_left_hand/0729",
        help="right robot + left human",
    )
    p.add_argument(
        "--frame_stride",
        type=int,
        default=20,
        help="Sample every Nth synced frame within each demo (default: 20, light on RAM)",
    )
    p.add_argument(
        "--max_frames_per_type",
        type=int,
        default= 0,
        help="Cap samples per dataset type (default 60; 0 = no cap)",
    )
    p.add_argument(
        "--max_demos",
        type=int,
        default=2,
        help="Per dataset type (default 2; 0=all). Keep small on ~16GB machines.",
    )
    p.add_argument("--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    p.add_argument("--max_skew_s", type=float, default=0.05)
    p.add_argument(
        "--jpeg_in_ram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store synced frames as JPEG bytes (much smaller than raw RGB)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--cpu_threads",
        type=int,
        default=2,
        help="Cap torch/OMP threads to avoid CPU thrash (default 2)",
    )
    p.add_argument(
        "--extractor",
        type=str,
        default="resnet_gap",
        choices=sorted(EXTRACTOR_REGISTRY),
        help="Which BackboneFeatureExtractor to use (always bird-view only)",
    )
    p.add_argument("--backbone", type=str, default="resnet18", help="Passed to model build()")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="Shared random_state for PCA / t-SNE / UMAP")
    p.add_argument(
        "--out_dir",
        type=str,
        default="recording/data_analysis/outputs/resnet_tsne",
    )
    return p.parse_args()


def _count_refs(refs: list[FrameRef]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in refs:
        out[r.dataset_type] = out.get(r.dataset_type, 0) + 1
    return out


def main() -> None:
    args = parse_args()

    # Keep CPU fan-out low so the machine stays responsive while decoding.
    n_threads = max(1, int(args.cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    max_demos = None if int(args.max_demos) <= 0 else int(args.max_demos)
    max_per_type = None if int(args.max_frames_per_type) <= 0 else int(args.max_frames_per_type)
    k = int(args.num_queries)

    teleop_root = resolve_path(args.teleop_root)
    human_root = resolve_path(args.human_root)
    mixed_lr_root = resolve_path(args.mixed_lr_root)
    mixed_rl_root = resolve_path(args.mixed_rl_root)

    sync_root = REPO_ROOT / "recording" / "data_analysis" / "m-synced-csvs"
    robot_sync = sync_root / f"{teleop_root.name}_robot_tsne"
    human_sync = sync_root / f"{human_root.name}_human_tsne"
    mixed_lr_sync = sync_root / f"{mixed_lr_root.parent.name}_{mixed_lr_root.name}_mixed_tsne"
    mixed_rl_sync = sync_root / f"{mixed_rl_root.parent.name}_{mixed_rl_root.name}_mixed_tsne"

    print(f"device={device} cpu_threads={n_threads}", flush=True)
    print(
        f"frame_stride={args.frame_stride} max_demos={max_demos} "
        f"max_frames_per_type={max_per_type} camera=bird (wrists unused)",
        flush=True,
    )
    print(f"extractor={args.extractor}", flush=True)

    # ---- datasets (JPEG-in-RAM keeps video storage small; we never cache float images) ----
    robot_eef = resolve_robot_eef_dir(teleop_root, None)
    human_pose = resolve_human_pose_dir(human_root, None)
    build_robot_sync_csvs(teleop_root, robot_sync, robot_eef, float(args.max_skew_s), max_demos)
    build_human_sync_csvs(human_root, human_sync, human_pose, float(args.max_skew_s), max_demos)

    robot_ds = RobotEpisodeDataset(
        bird_vids_dir=teleop_root / "bird-realsense-data" / "mp4",
        front_vids_dir=teleop_root / "front-realsense-data" / "mp4",
        left_arm_vids_dir=teleop_root / "aloha-data" / "left" / "mp4",
        right_arm_vids_dir=teleop_root / "aloha-data" / "right" / "mp4",
        left_joint_data_dir=teleop_root / "joint-data" / "left" / "position",
        right_joint_data_dir=teleop_root / "joint-data" / "right" / "position",
        eef_pose_data_dir=robot_eef,
        sync_csv_dir=robot_sync,
        num_queries=k,
        max_demos=max_demos,
        jpeg_in_ram=bool(args.jpeg_in_ram),
    )
    human_ds = HumanEpisodeDataset(
        bird_vids_dir=human_root / "bird-realsense-data" / "mp4",
        front_vids_dir=human_root / "front-realsense-data" / "mp4",
        pose_npz_dir=human_pose,
        sync_csv_dir=human_sync,
        num_queries=k,
        max_demos=max_demos,
        jpeg_in_ram=bool(args.jpeg_in_ram),
    )

    mixed_lr_ds: MixedEpisodeDataset | None = None
    mixed_rl_ds: MixedEpisodeDataset | None = None
    ckpts = [resolve_path(c) for c in args.checkpoints]
    model_pkgs = list(args.model_pkgs) if args.model_pkgs else [infer_model_pkg(c) for c in ckpts]
    if len(model_pkgs) != len(ckpts):
        raise ValueError("--model_pkgs length must match --checkpoints")
    any_mixed_pkg = any(p.startswith("MixedEmbodiment") for p in model_pkgs)

    if any_mixed_pkg:
        eef_staging_root = sync_root / "_eef_remapped"
        mixed_children = []
        for mixed_root, mixed_sync, label in (
            (mixed_lr_root, mixed_lr_sync, "left_robot_right_hand"),
            (mixed_rl_root, mixed_rl_sync, "right_robot_left_hand"),
        ):
            preset = _infer_preset(mixed_root)
            if preset is None:
                raise ValueError(f"Cannot infer robot/hand sides from {mixed_root}")
            robot_side = preset["robot_side"]
            hand_side = preset["hand_side"]
            pose_dir = _resolve_pose_dir(mixed_root, None)
            eef_dir = remap_eef_dir_to_session_ids(
                mixed_root,
                _resolve_eef_dir(mixed_root, None),
                eef_staging_root,
            )
            print(f"Mixed {label}: robot_side={robot_side} hand_side={hand_side}", flush=True)
            build_mixed_sync_csvs(
                mixed_root,
                mixed_sync,
                robot_side=robot_side,
                hand_side=hand_side,
                pose_dir=pose_dir,
                eef_dir=eef_dir,
                max_skew_s=float(args.max_skew_s),
                max_demos=max_demos,
            )
            child = MixedEpisodeDataset(
                bird_vids_dir=mixed_root / "bird-realsense-data" / "mp4",
                wrist_vids_dir=mixed_root / "aloha-data" / robot_side / "mp4",
                joint_data_dir=mixed_root / "joint-data" / robot_side / "position",
                hand_pose_npz_dir=pose_dir,
                eef_pose_data_dir=eef_dir,
                sync_csv_dir=mixed_sync,
                robot_side=robot_side,
                hand_side=hand_side,
                num_queries=k,
                max_demos=max_demos,
                jpeg_in_ram=bool(args.jpeg_in_ram),
            )
            mixed_children.append(child)
        mixed_lr_ds, mixed_rl_ds = mixed_children

    refs_all = collect_strided_refs(
        robot_ds=robot_ds,
        human_ds=human_ds,
        mixed_lr=mixed_lr_ds if any_mixed_pkg else None,
        mixed_rl=mixed_rl_ds if any_mixed_pkg else None,
        frame_stride=int(args.frame_stride),
        max_frames_per_type=max_per_type,
        include_mixed=any_mixed_pkg,
    )
    refs_combined = [r for r in refs_all if r.dataset_type in DATASET_TYPES_COMBINED]
    refs_mixed: list[FrameRef] | None = refs_all if any_mixed_pkg else None

    print(f"Combined frame refs: {len(refs_combined)} {_count_refs(refs_combined)}", flush=True)
    if refs_mixed is not None:
        print(f"Mixed frame refs: {len(refs_mixed)} {_count_refs(refs_mixed)}", flush=True)
    print("Streaming extract (1 frame in RAM at a time)...", flush=True)

    summary: dict[str, Any] = {
        "frame_stride": int(args.frame_stride),
        "max_demos": max_demos,
        "max_frames_per_type": max_per_type,
        "extractor": args.extractor,
        "cameras": "bird",
        "perplexity": float(args.perplexity),
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "seed": int(args.seed),
        "checkpoints": [],
    }

    for ckpt, pkg in zip(ckpts, model_pkgs):
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        include_mixed = pkg.startswith("MixedEmbodiment")
        refs = refs_mixed if include_mixed else refs_combined
        assert refs is not None
        dataset_order = dataset_types_for_pkg(pkg)
        types = [r.dataset_type for r in refs]
        progress = np.asarray([r.progress for r in refs], dtype=np.float32)
        demo_idx = np.asarray([r.demo_idx for r in refs], dtype=np.int32)
        start_in_ep = np.asarray([r.start_in_ep for r in refs], dtype=np.int32)
        seed = int(args.seed)

        print(f"\n=== {pkg} :: {ckpt.name} ===", flush=True)
        print(f"  frames={len(refs)} types={_count_refs(refs)}", flush=True)

        build = load_build_fn(pkg)
        model = build(BuildArgs(k, backbone=args.backbone)).to(device)
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state)
        model.eval()
        del state

        extractor = build_extractor(
            args.extractor,
            model,
            device=device,
        )

        feat_arr = extract_embeddings_streaming(
            refs,
            extractor,
            robot_ds=robot_ds,
            human_ds=human_ds,
            mixed_lr=mixed_lr_ds,
            mixed_rl=mixed_rl_ds,
        )
        print(f"  feature matrix: {feat_arr.shape}", flush=True)

        pca_xy = run_pca(feat_arr, random_state=seed)
        tsne_xy = run_tsne(
            feat_arr,
            perplexity=float(args.perplexity),
            random_state=seed,
        )
        umap_xy = run_umap(
            feat_arr,
            n_neighbors=int(args.umap_n_neighbors),
            min_dist=float(args.umap_min_dist),
            random_state=seed,
        )

        ckpt_dir = out_dir / f"{pkg}_{ckpt.stem}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            ckpt_dir / "features.npz",
            features=feat_arr,
            dataset_type=np.asarray(types),
            progress=progress,
            demo_idx=demo_idx,
            start_in_ep=start_in_ep,
            checkpoint=str(ckpt),
            model_pkg=pkg,
            extractor=args.extractor,
            cameras="bird",
        )
        np.savez_compressed(
            ckpt_dir / "embeddings_2d.npz",
            pca=pca_xy,
            tsne=tsne_xy,
            umap=umap_xy,
            dataset_type=np.asarray(types),
            progress=progress,
            demo_idx=demo_idx,
            start_in_ep=start_in_ep,
            perplexity=np.float32(args.perplexity),
            umap_n_neighbors=np.int32(args.umap_n_neighbors),
            umap_min_dist=np.float32(args.umap_min_dist),
            random_state=np.int32(seed),
        )

        for method, xy, xlab, ylab in (
            ("pca", pca_xy, "PC1", "PC2"),
            ("tsne", tsne_xy, "t-SNE 1", "t-SNE 2"),
            ("umap", umap_xy, "UMAP 1", "UMAP 2"),
        ):
            fig_path = ckpt_dir / f"{args.extractor}_{method}_by_dataset.png"
            scatter_by_dataset(
                xy,
                types,
                dataset_order=dataset_order,
                title=f"{ckpt.stem} | {args.extractor} {method.upper()} (by dataset)",
                out_path=fig_path,
                xlabel=xlab,
                ylabel=ylab,
            )
            print(f"  wrote {fig_path}", flush=True)

        summary["checkpoints"].append(
            {
                "checkpoint": str(ckpt),
                "model_pkg": pkg,
                "out_dir": str(ckpt_dir),
                "n_frames": int(feat_arr.shape[0]),
                "feat_dim": int(feat_arr.shape[1]),
                "counts": _count_refs(refs),
            }
        )

        del model, extractor, feat_arr, pca_xy, tsne_xy, umap_xy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote PCA / t-SNE / UMAP outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
