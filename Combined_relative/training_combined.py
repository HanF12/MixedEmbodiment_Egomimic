"""
Mixed human + robot ACT training with relative pose actions (EgoMimic-aligned).

- Pose = xyz+gripper only (8D); rot6d dropped at load
- Pose actions are relative to the first observation in the chunk:
    pose_actions[k] = pose[t+k] - pose[t]
  At inference: abs[k] = anchor_pose[t] + denorm(pred[k])
- Joint actions remain absolute
- Shared pose head (human primary + robot aux); robot-only joint head
- Modality routing (no embodiment embedding)
- Schedule: each epoch = one full pass over the longer modality's demo loader
  (shorter modality recycled); batch 8, lr 1e-5, L1, kl=10, hand_lambda
- Horizon K kept at 45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Combined_relative.config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_HAND_LAMBDA,
    DEFAULT_HUMAN_DATA_ROOT,
    DEFAULT_KL_WEIGHT,
    DEFAULT_LR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_QUERIES,
    DEFAULT_RECON_LOSS,
    DEFAULT_ROBOT_DATA_ROOT,
    DEFAULT_WEIGHT_DECAY,
    EMBODIMENT_HUMAN,
    EMBODIMENT_ROBOT,
    HUMAN_POSE_RELDIR,
    MODEL_CAMERA_NAMES,
    POSE_DIM,
    ROBOT_EEF_RELDIR,
    ROBOT_JOINT_DIM,
    build_run_metadata,
    default_run_name,
    save_run_metadata,
)
from Combined_relative.core import build, kl_divergence  # noqa: E402
from Combined_relative.data_synchronization import (  # noqa: E402
    synchronize_human_hands,
    synchronize_robot_bimanual,
)
from Combined_relative.dataloader_human import HumanEpisodeDataset, collate_homogeneous  # noqa: E402
from Combined_relative.dataloader_robot import RobotEpisodeDataset  # noqa: E402
from Combined_relative.dataloader_utils import (  # noqa: E402
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    demo_id_from_pose_npz,
    demo_id_from_robot_eef_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combined-relative mixed human+robot ACT training (relative pose actions)"
    )
    p.add_argument(
        "--robot_data_root",
        type=str,
        default=str(DEFAULT_ROBOT_DATA_ROOT),
        help=f"Robot teleop root (default: {DEFAULT_ROBOT_DATA_ROOT})",
    )
    p.add_argument(
        "--human_data_root",
        type=str,
        default=str(DEFAULT_HUMAN_DATA_ROOT),
        help=f"Human hands root (default: {DEFAULT_HUMAN_DATA_ROOT})",
    )
    p.add_argument("--robot_sync_dir", type=str, default=None)
    p.add_argument("--human_sync_dir", type=str, default=None)
    p.add_argument(
        "--robot_eef_dir",
        type=str,
        default=None,
        help=f"Robot EEF NPZ dir (default: <robot_data_root>/{ROBOT_EEF_RELDIR})",
    )
    p.add_argument(
        "--human_pose_dir",
        type=str,
        default=None,
        help=f"Human pose NPZ dir (default: <human_data_root>/{HUMAN_POSE_RELDIR})",
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help=(
            f"Number of epochs (default {DEFAULT_NUM_EPOCHS}). "
            "Each epoch is one full pass over the longer modality's demo loader "
            "(max batches); the shorter modality is recycled."
        ),
    )
    p.add_argument("-b", "--batch", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("-q", "--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    p.add_argument("-g", "--gpu_number", type=int, default=0)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max_skew_s", type=float, default=0.050)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--max_demos",
        type=int,
        default=None,
        help="Use only the first N demos per modality (sorted demo IDs). Alias of --first_n.",
    )
    p.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="Same as --max_demos: use first N demos (sorted) for both robot and human.",
    )
    p.add_argument(
        "--robot_first_n",
        type=int,
        default=None,
        help="Use only the first N robot demos (sorted demo IDs). Overrides --first_n/--max_demos for robot.",
    )
    p.add_argument(
        "--human_first_n",
        type=int,
        default=None,
        help="Use only the first N human demos (sorted demo IDs). Overrides --first_n/--max_demos for human.",
    )
    p.add_argument("--resize_factor", type=float, default=1.0)
    p.add_argument("--max_sync_rows", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_every_epochs", type=int, default=200)
    p.add_argument("--pose_loss_weight", type=float, default=1.0)
    p.add_argument("--joint_loss_weight", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=DEFAULT_KL_WEIGHT)
    p.add_argument("--hand_lambda", type=float, default=DEFAULT_HAND_LAMBDA, help="EgoMimic human loss scale")
    p.add_argument(
        "--reconstruction_loss",
        type=str,
        choices=("mse", "l1"),
        default=DEFAULT_RECON_LOSS,
    )
    p.add_argument(
        "--joint_modality_update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate human+robot losses then one optimizer step (default, EgoMimic). "
        "Shorter modality demos are recycled so both sides contribute every step. "
        "Use --no-joint_modality_update for alternating single-modality steps.",
    )
    p.add_argument(
        "--no_front_camera",
        action="store_true",
        help="Disable front camera for training: zero front RGB slot and set camera_mask[front]=0 "
        "(bird+wrists for robot; bird-only for human). Front is still used only for sync CSVs.",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="combined-mixed-act")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Sync (if roots given), load one batch per available modality, one train step each.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Ultra-light validation: 1 demo/modality, tiny frames, CPU, dry_run.",
    )
    p.add_argument(
        "--synthetic_robot",
        action="store_true",
        help="If robot data is missing, run a synthetic robot batch to exercise the robot heads.",
    )
    return p.parse_args()


def resolve_demo_caps(cli: argparse.Namespace) -> argparse.Namespace:
    """
    Resolve first-N demo caps.

    Priority per modality:
      --robot_first_n / --human_first_n  >  --first_n  >  --max_demos  >  None (all)
    """
    shared = cli.first_n if cli.first_n is not None else cli.max_demos
    # Keep max_demos in sync for smoke / legacy prints
    if cli.max_demos is None and cli.first_n is not None:
        cli.max_demos = cli.first_n
    elif cli.first_n is None and cli.max_demos is not None:
        cli.first_n = cli.max_demos

    cli.robot_demo_cap = cli.robot_first_n if cli.robot_first_n is not None else shared
    cli.human_demo_cap = cli.human_first_n if cli.human_first_n is not None else shared
    return cli


def apply_smoke_defaults(cli: argparse.Namespace) -> argparse.Namespace:
    if not cli.smoke:
        return cli
    cli.dry_run = True
    cli.cpu = True
    cli.wandb = False
    if cli.max_demos is None and cli.first_n is None and cli.robot_first_n is None and cli.human_first_n is None:
        cli.max_demos = 1
        cli.first_n = 1
    if cli.batch > 1:
        cli.batch = 1
    if cli.num_workers != 0:
        cli.num_workers = 0
    if cli.num_queries > 8:
        cli.num_queries = 8
    if cli.resize_factor >= 1.0:
        cli.resize_factor = 0.25
    if cli.max_sync_rows is None:
        cli.max_sync_rows = 32
    if cli.epochs > 1 and not cli.dry_run:
        cli.epochs = 1
    if cli.robot_data_root is None:
        cli.synthetic_robot = True
    cli = resolve_demo_caps(cli)
    print(
        "SMOKE MODE: "
        f"robot_first_n={cli.robot_demo_cap} human_first_n={cli.human_demo_cap} "
        f"batch={cli.batch} workers={cli.num_workers} "
        f"q={cli.num_queries} resize={cli.resize_factor} max_sync_rows={cli.max_sync_rows} "
        f"cpu=True dry_run=True synthetic_robot={cli.synthetic_robot}"
    )
    return cli


class Args:
    def __init__(self, num_queries: int):
        self.num_queries = num_queries
        self.camera_names = list(MODEL_CAMERA_NAMES)
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        self.position_embedding = "sine"
        self.backbone = "resnet18"
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False


def resolve_human_pose_dir(human_root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    preferred = (human_root / HUMAN_POSE_RELDIR).resolve()
    if preferred.is_dir() and any(preferred.glob("*.npz")):
        return preferred
    fallback = (human_root / "bird-realsense-data" / "combined_npz").resolve()
    if fallback.is_dir() and any(fallback.glob("*.npz")):
        print(
            f"WARNING: default human pose dir missing/empty ({preferred}); "
            f"falling back to {fallback}"
        )
        return fallback
    raise FileNotFoundError(
        f"Human pose NPZ dir not found. Expected default {preferred} "
        f"(or override with --human_pose_dir)."
    )


def resolve_robot_eef_dir(robot_root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    preferred = (robot_root / ROBOT_EEF_RELDIR).resolve()
    if preferred.is_dir() and any(preferred.glob("*.npz")):
        return preferred
    raise FileNotFoundError(
        f"Robot EEF NPZ dir not found. Expected default {preferred} "
        f"(or override with --robot_eef_dir)."
    )


def build_robot_sync_csvs(
    data_root: Path,
    sync_dir: Path,
    eef_dir: Path,
    max_skew_s: float,
    max_demos: int | None,
) -> None:
    bird = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "bird-realsense-data" / "npy").glob("*.npy"))}
    front = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "front-realsense-data" / "npy").glob("*.npy"))}
    left_c = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "aloha-data" / "left" / "npy").glob("*.npy"))}
    right_c = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "aloha-data" / "right" / "npy").glob("*.npy"))}
    left_j = {
        demo_id_from_joint_npy(p, prefix="joint_timestamp_"): p
        for p in sorted((data_root / "joint-data" / "left" / "time").glob("*.npy"))
    }
    right_j = {
        demo_id_from_joint_npy(p, prefix="joint_timestamp_"): p
        for p in sorted((data_root / "joint-data" / "right" / "time").glob("*.npy"))
    }
    eef = {demo_id_from_robot_eef_npz(p): p for p in sorted(eef_dir.glob("*.npz"))}

    base_ids = sorted(set(bird) & set(front) & set(left_c) & set(right_c) & set(left_j) & set(right_j))
    ids = sorted(set(base_ids) & set(eef))
    skipped_no_eef = sorted(set(base_ids) - set(eef))
    for demo_id in skipped_no_eef:
        print(f"WARNING: skip robot {demo_id} - missing EEF pose under {eef_dir}")

    if max_demos is not None and max_demos > 0:
        ids = ids[:max_demos]
    if not ids:
        raise FileNotFoundError(
            f"No complete robot demos under {data_root} with EEF in {eef_dir}. "
            f"base_complete={len(base_ids)} eef={len(eef)}"
        )
    sync_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building robot sync for {len(ids)} demos -> {sync_dir}")
    for demo_id in ids:
        eef_npz = np.load(eef[demo_id])
        for key in ("timestamps", "pose", "valid_pos", "valid_open"):
            if key not in eef_npz.files:
                raise KeyError(f"{eef[demo_id].name} missing required key '{key}'")
        synchronize_robot_bimanual(
            np.load(left_j[demo_id]),
            np.load(right_j[demo_id]),
            np.load(left_c[demo_id]),
            np.load(right_c[demo_id]),
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            sync_dir / f"{demo_id}.csv",
            eef_ts=eef_npz["timestamps"],
            max_skew_s=max_skew_s,
            debug=False,
            valid_pos=eef_npz["valid_pos"],
            valid_open=eef_npz["valid_open"],
            require_full_eef_pose=True,
        )


def build_human_sync_csvs(
    data_root: Path,
    sync_dir: Path,
    pose_dir: Path,
    max_skew_s: float,
    max_demos: int | None,
) -> None:
    bird = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "bird-realsense-data" / "npy").glob("*.npy"))}
    front = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "front-realsense-data" / "npy").glob("*.npy"))}
    pose = {demo_id_from_pose_npz(p): p for p in sorted(pose_dir.glob("*.npz"))}
    ids = sorted(set(bird) & set(front) & set(pose))
    if max_demos is not None and max_demos > 0:
        ids = ids[:max_demos]
    if not ids:
        raise FileNotFoundError(
            f"No complete human demos under {data_root}. "
            f"Need bird/front npy timestamps and pose NPZs in {pose_dir}."
        )
    sync_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building human sync for {len(ids)} demos -> {sync_dir} (pose_dir={pose_dir})")
    for demo_id in ids:
        pose_npz = np.load(pose[demo_id])
        for key in ("valid_pos", "valid_open"):
            if key not in pose_npz.files:
                raise KeyError(f"{pose[demo_id].name} missing required validity key '{key}'")
        synchronize_human_hands(
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            pose_npz["timestamps"],
            sync_dir / f"{demo_id}.csv",
            max_skew_s=max_skew_s,
            debug=False,
            valid_pos=pose_npz["valid_pos"],
            valid_open=pose_npz["valid_open"],
            require_full_pose=True,
        )


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_homogeneous,
    )


def synthetic_robot_batch(batch_size: int, num_queries: int, h: int = 120, w: int = 160, device=None) -> dict:
    device = device or torch.device("cpu")
    return {
        "embodiment": EMBODIMENT_ROBOT,
        "images": torch.randn(batch_size, 4, 3, h, w, device=device),
        "camera_mask": torch.ones(batch_size, 4, device=device),
        "pose_state": torch.randn(batch_size, POSE_DIM, device=device),
        "pose_actions": torch.randn(batch_size, num_queries, POSE_DIM, device=device),
        "joint_state": torch.randn(batch_size, ROBOT_JOINT_DIM, device=device),
        "joint_actions": torch.randn(batch_size, num_queries, ROBOT_JOINT_DIM, device=device),
        "has_joint_target": True,
        "is_pad": torch.zeros(batch_size, num_queries, dtype=torch.bool, device=device),
    }


def synthetic_human_batch(batch_size: int, num_queries: int, h: int = 120, w: int = 160, device=None) -> dict:
    device = device or torch.device("cpu")
    return {
        "embodiment": EMBODIMENT_HUMAN,
        "images": torch.randn(batch_size, 4, 3, h, w, device=device),
        "camera_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], device=device).expand(batch_size, -1).clone(),
        "pose_state": torch.randn(batch_size, POSE_DIM, device=device),
        "pose_actions": torch.randn(batch_size, num_queries, POSE_DIM, device=device),
        "joint_state": torch.zeros(batch_size, ROBOT_JOINT_DIM, device=device),
        "joint_actions": torch.zeros(batch_size, num_queries, ROBOT_JOINT_DIM, device=device),
        "has_joint_target": False,
        "is_pad": torch.zeros(batch_size, num_queries, dtype=torch.bool, device=device),
    }


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor, *, kind: str) -> torch.Tensor:
    """Mean over valid scalar elements (not just timesteps), so 20D vs 14D are comparable."""
    if kind == "mse":
        err = F.mse_loss(pred, target, reduction="none")
    elif kind == "l1":
        err = F.l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction_loss={kind}")
    valid = (~is_pad).unsqueeze(-1).expand_as(err)
    if not valid.any():
        return pred.new_zeros(())
    return err[valid].mean()


def compute_batch_losses(
    model,
    batch,
    device,
    *,
    pose_w,
    joint_w,
    kl_w,
    recon_kind,
    hand_lambda,
) -> dict:
    """
    EgoMimic-style losses:
      human: hand_lambda * (pose_recon + kl)
      robot: pose_recon (aux shared head) + joint_recon + kl
    """
    pose_state = batch["pose_state"].to(device)
    joint_state = batch["joint_state"].to(device)
    images = batch["images"].to(device)
    pose_actions = batch["pose_actions"].to(device)
    joint_actions = batch["joint_actions"].to(device)
    is_pad = batch["is_pad"].to(device)
    camera_mask = batch["camera_mask"].to(device)
    embodiment = int(batch["embodiment"])
    has_joint = bool(batch["has_joint_target"])

    out = model(
        pose_state=pose_state,
        images=images,
        embodiment=embodiment,
        joint_state=joint_state,
        camera_mask=camera_mask,
        pose_actions=pose_actions,
        joint_actions=joint_actions,
        has_joint_target=has_joint,
        is_pad=is_pad,
    )
    pose_pred = out["pose_pred"]
    joint_pred = out["joint_pred"]
    mu, logvar = out["mu"], out["logvar"]
    total_kld, *_ = kl_divergence(mu, logvar)
    kld = total_kld[0]

    pose_loss = masked_recon_loss(pose_pred, pose_actions, is_pad, kind=recon_kind)
    if has_joint:
        if joint_pred is None:
            raise RuntimeError("Robot batch missing joint_pred")
        joint_loss = masked_recon_loss(joint_pred, joint_actions, is_pad, kind=recon_kind)
        # Robot: joint + aux pose + KL (no hand_lambda)
        loss = pose_w * pose_loss + joint_w * joint_loss + kl_w * kld
        joint_pred_shape = tuple(joint_pred.shape)
    else:
        joint_loss = pose_pred.new_zeros(())
        # Human: scale pose + KL by hand_lambda (EgoMimic)
        loss = float(hand_lambda) * (pose_w * pose_loss + kl_w * kld)
        joint_pred_shape = None

    name = "robot" if embodiment == EMBODIMENT_ROBOT else "human"
    return {
        "loss": loss,
        "pose_loss": pose_loss.detach(),
        "joint_loss": joint_loss.detach(),
        "kld": kld.detach(),
        "embodiment": name,
        "has_joint_target": has_joint,
        "pose_pred_shape": tuple(pose_pred.shape),
        "joint_pred_shape": joint_pred_shape,
    }


def train_step_single(model, optimizer, scaler, batch, device, use_amp: bool, loss_cfg: dict) -> dict:
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        stats = compute_batch_losses(model, batch, device, **loss_cfg)
        loss = stats["loss"]
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach().cpu()),
        "pose_loss": float(stats["pose_loss"].cpu()),
        "joint_loss": float(stats["joint_loss"].cpu()),
        "kld": float(stats["kld"].cpu()),
        "embodiment": stats["embodiment"],
        "pose_pred_shape": stats["pose_pred_shape"],
        "joint_pred_shape": stats["joint_pred_shape"],  # None for human
    }


def train_step_joint_modalities(
    model,
    optimizer,
    scaler,
    robot_batch,
    human_batch,
    device,
    use_amp: bool,
    loss_cfg: dict,
) -> dict:
    """One optimizer step from averaged human+robot losses (EgoMimic-style)."""
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        r = compute_batch_losses(model, robot_batch, device, **loss_cfg)
        h = compute_batch_losses(model, human_batch, device, **loss_cfg)
        # Average so gradient magnitude does not double vs single-modality step
        loss = 0.5 * (r["loss"] + h["loss"])
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach().cpu()),
        "robot_pose_loss": float(r["pose_loss"].cpu()),
        "robot_joint_loss": float(r["joint_loss"].cpu()),
        "human_pose_loss": float(h["pose_loss"].cpu()),
        "kld": float(0.5 * (r["kld"] + h["kld"]).cpu()),
        "robot_pose_pred_shape": r["pose_pred_shape"],
        "robot_joint_pred_shape": r["joint_pred_shape"],
        "human_pose_pred_shape": h["pose_pred_shape"],
    }


def next_batch_recycling(loader_iter, loader):
    """
    Fetch the next batch; if the loader is exhausted, restart it (recycle demos).

    Used so the shorter modality keeps contributing paired updates for the full
    epoch length max(len(robot_loader), len(human_loader)).
    """
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def steps_per_epoch_from_loaders(robot_loader, human_loader) -> int:
    """One epoch = one full demo-set pass over the longer modality loader."""
    lengths = []
    if robot_loader is not None:
        lengths.append(len(robot_loader))
    if human_loader is not None:
        lengths.append(len(human_loader))
    if not lengths:
        return 1
    return max(1, max(lengths))


def main() -> None:
    cli = parse_args()
    if cli.smoke:
        cli = apply_smoke_defaults(cli)
    else:
        cli = resolve_demo_caps(cli)
    pkg = Path(__file__).resolve().parent

    if cli.robot_demo_cap is not None or cli.human_demo_cap is not None:
        print(
            f"Demo caps: robot_first_n={cli.robot_demo_cap} human_first_n={cli.human_demo_cap} "
            f"(sorted demo IDs; None = all)"
        )

    robot_root = Path(cli.robot_data_root).expanduser().resolve() if cli.robot_data_root else None
    human_root = Path(cli.human_data_root).expanduser().resolve() if cli.human_data_root else None

    if robot_root is None and human_root is None and not cli.synthetic_robot:
        raise ValueError("Provide --robot_data_root and/or --human_data_root (or --smoke/--synthetic_robot).")

    robot_sync = (
        Path(cli.robot_sync_dir).expanduser().resolve()
        if cli.robot_sync_dir
        else (pkg / "m-synced-csvs" / f"{robot_root.name if robot_root else 'none'}_robot")
    )
    human_sync = (
        Path(cli.human_sync_dir).expanduser().resolve()
        if cli.human_sync_dir
        else (pkg / "m-synced-csvs" / f"{human_root.name if human_root else 'none'}_human")
    )

    robot_eef_dir = resolve_robot_eef_dir(robot_root, cli.robot_eef_dir) if robot_root else None
    human_pose_dir = resolve_human_pose_dir(human_root, cli.human_pose_dir) if human_root else None
    if robot_eef_dir is not None:
        print(f"Robot EEF NPZ dir (default {ROBOT_EEF_RELDIR}): {robot_eef_dir}")
    if human_pose_dir is not None:
        print(f"Human pose NPZ dir (default {HUMAN_POSE_RELDIR}): {human_pose_dir}")

    weights_root = Path(cli.output_dir).expanduser().resolve() if cli.output_dir else (pkg / "weights")
    run_name = cli.run_name or default_run_name()
    if cli.smoke and cli.run_name is None:
        run_name = f"{run_name}_smoke"
    output_dir = weights_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_cfg = {
        "pose_w": float(cli.pose_loss_weight),
        "joint_w": float(cli.joint_loss_weight),
        "kl_w": float(cli.kl_weight),
        "recon_kind": str(cli.reconstruction_loss),
        "hand_lambda": float(cli.hand_lambda),
    }

    robot_ds = None
    human_ds = None

    if robot_root is not None:
        if not robot_root.exists():
            raise FileNotFoundError(robot_root)
        if robot_eef_dir is None or not robot_eef_dir.exists():
            raise FileNotFoundError(f"Robot EEF dir not found: {robot_eef_dir}")
        build_robot_sync_csvs(robot_root, robot_sync, robot_eef_dir, cli.max_skew_s, cli.robot_demo_cap)
        robot_ds = RobotEpisodeDataset(
            bird_vids_dir=robot_root / "bird-realsense-data" / "mp4",
            front_vids_dir=robot_root / "front-realsense-data" / "mp4",
            left_arm_vids_dir=robot_root / "aloha-data" / "left" / "mp4",
            right_arm_vids_dir=robot_root / "aloha-data" / "right" / "mp4",
            left_joint_data_dir=robot_root / "joint-data" / "left" / "position",
            right_joint_data_dir=robot_root / "joint-data" / "right" / "position",
            eef_pose_data_dir=robot_eef_dir,
            sync_csv_dir=robot_sync,
            num_queries=cli.num_queries,
            max_demos=cli.robot_demo_cap,
            resize_factor=cli.resize_factor,
            max_sync_rows=cli.max_sync_rows,
            disable_front_camera=cli.no_front_camera,
        )
        np.savez(
            output_dir / "normalization_stats_robot.npz",
            joint_mean=robot_ds.joint_mean.numpy(),
            joint_std=robot_ds.joint_std.numpy(),
            # Primary action stats: relative pose (chunk-anchored deltas)
            eef_pose_mean=robot_ds.eef_mean.numpy(),
            eef_pose_std=robot_ds.eef_std.numpy(),
            eef_pose_rel_mean=robot_ds.eef_rel_mean.numpy(),
            eef_pose_rel_std=robot_ds.eef_rel_std.numpy(),
            # Absolute pose stats for proprio / inference anchoring
            eef_pose_abs_mean=robot_ds.eef_abs_mean.numpy(),
            eef_pose_abs_std=robot_ds.eef_abs_std.numpy(),
            pose_action_space=np.asarray("relative_to_chunk_anchor"),
            # backward-compatible aliases
            qpos_mean=robot_ds.joint_mean.numpy(),
            qpos_std=robot_ds.joint_std.numpy(),
        )

    if human_root is not None:
        if not human_root.exists():
            raise FileNotFoundError(human_root)
        if human_pose_dir is None or not human_pose_dir.exists():
            raise FileNotFoundError(f"Human pose dir not found: {human_pose_dir}")
        build_human_sync_csvs(human_root, human_sync, human_pose_dir, cli.max_skew_s, cli.human_demo_cap)
        human_ds = HumanEpisodeDataset(
            bird_vids_dir=human_root / "bird-realsense-data" / "mp4",
            front_vids_dir=human_root / "front-realsense-data" / "mp4",
            pose_npz_dir=human_pose_dir,
            sync_csv_dir=human_sync,
            num_queries=cli.num_queries,
            max_demos=cli.human_demo_cap,
            resize_factor=cli.resize_factor,
            max_sync_rows=cli.max_sync_rows,
            disable_front_camera=cli.no_front_camera,
        )
        np.savez(
            output_dir / "normalization_stats_human.npz",
            # Primary action stats: relative pose (chunk-anchored deltas)
            pose_mean=human_ds.pose_mean.numpy(),
            pose_std=human_ds.pose_std.numpy(),
            pose_rel_mean=human_ds.pose_rel_mean.numpy(),
            pose_rel_std=human_ds.pose_rel_std.numpy(),
            # Absolute pose stats for proprio / inference anchoring
            pose_abs_mean=human_ds.pose_abs_mean.numpy(),
            pose_abs_std=human_ds.pose_abs_std.numpy(),
            pose_action_space=np.asarray("relative_to_chunk_anchor"),
        )

    meta = build_run_metadata(
        robot_data_root=robot_root,
        human_data_root=human_root,
        robot_sync_dir=robot_sync if robot_root else None,
        human_sync_dir=human_sync if human_root else None,
        num_queries=cli.num_queries,
        max_skew_s=cli.max_skew_s,
        robot_eef_dir=robot_eef_dir,
        pose_loss_weight=cli.pose_loss_weight,
        joint_loss_weight=cli.joint_loss_weight,
        kl_weight=cli.kl_weight,
        reconstruction_loss=cli.reconstruction_loss,
        joint_modality_update=cli.joint_modality_update,
        hand_lambda=cli.hand_lambda,
        num_epochs=cli.epochs,
        batch_size=cli.batch,
        lr=cli.lr,
        disable_front_camera=cli.no_front_camera,
    )
    if human_pose_dir is not None:
        meta["human_pose_dir"] = str(human_pose_dir)

    device = torch.device("cpu" if cli.cpu or not torch.cuda.is_available() else f"cuda:{cli.gpu_number}")
    print(f"Using device: {device}")
    model = build(Args(cli.num_queries)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    robot_loader = make_loader(robot_ds, cli.batch, cli.num_workers, shuffle=True) if robot_ds is not None else None
    human_loader = make_loader(human_ds, cli.batch, cli.num_workers, shuffle=True) if human_ds is not None else None
    steps_per_epoch = steps_per_epoch_from_loaders(robot_loader, human_loader)
    if cli.smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    meta["steps_per_epoch"] = int(steps_per_epoch)
    if robot_loader is not None:
        meta["robot_loader_batches"] = len(robot_loader)
        meta["robot_num_demos"] = len(robot_ds) if robot_ds is not None else None
    if human_loader is not None:
        meta["human_loader_batches"] = len(human_loader)
        meta["human_num_demos"] = len(human_ds) if human_ds is not None else None
    save_run_metadata(output_dir, meta)

    if cli.dry_run:
        print("--- Combined-relative dry run ---")
        if human_loader is not None:
            batch = next(iter(human_loader))
            stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
            print(f"Human step OK: {stats}")
        else:
            batch = synthetic_human_batch(cli.batch, cli.num_queries, device=device)
            stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
            print(f"Synthetic human step OK: {stats}")
        if robot_loader is not None:
            batch = next(iter(robot_loader))
            stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
            print(f"Robot step OK: {stats}")
        elif cli.synthetic_robot:
            batch = synthetic_robot_batch(cli.batch, cli.num_queries, device=device)
            stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
            print(f"Synthetic robot step OK: {stats}")
        # Verify shared pose head identity
        assert hasattr(model, "pose_action_head")
        print(
            f"Shared pose_action_head id={id(model.pose_action_head)} "
            f"joint_action_head id={id(model.joint_action_head)}"
        )
        print("Dry run passed.")
        return

    if robot_loader is None or human_loader is None:
        raise RuntimeError("Full training requires both --robot_data_root and --human_data_root.")

    wandb_run = None
    if cli.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed")
        wandb_run = wandb.init(
            project=cli.wandb_project,
            entity=cli.wandb_entity,
            name=cli.wandb_run_name,
            mode=cli.wandb_mode,
            config=vars(cli),
        )

    best = float("inf")
    step = 0
    print(
        f"Training schedule: {cli.epochs} epochs x {steps_per_epoch} steps/epoch "
        f"(= max(robot_batches={len(robot_loader)}, human_batches={len(human_loader)}); "
        f"shorter modality recycled), "
        f"batch={cli.batch}, lr={cli.lr}, recon={cli.reconstruction_loss}, "
        f"kl_weight={cli.kl_weight}, hand_lambda={cli.hand_lambda}, K={cli.num_queries}"
    )
    for epoch in range(cli.epochs):
        model.train()
        robot_iter = iter(robot_loader)
        human_iter = iter(human_loader)
        running = {
            "robot_pose": 0.0,
            "robot_joint": 0.0,
            "human_pose": 0.0,
            "n_r": 0,
            "n_h": 0,
            "n_joint": 0,
        }
        loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{cli.epochs}", unit="step")
        for _ in loop:
            robot_batch, robot_iter = next_batch_recycling(robot_iter, robot_loader)
            human_batch, human_iter = next_batch_recycling(human_iter, human_loader)

            if cli.joint_modality_update:
                stats = train_step_joint_modalities(
                    model, optimizer, scaler, robot_batch, human_batch, device, use_amp, loss_cfg
                )
                step += 1
                running["robot_pose"] += stats["robot_pose_loss"]
                running["robot_joint"] += stats["robot_joint_loss"]
                running["human_pose"] += stats["human_pose_loss"]
                running["n_r"] += 1
                running["n_h"] += 1
                running["n_joint"] += 1
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/robot_pose_loss": stats["robot_pose_loss"],
                            "train/robot_joint_loss": stats["robot_joint_loss"],
                            "train/human_pose_loss": stats["human_pose_loss"],
                            "train/kl_loss": stats["kld"],
                            "train/total_loss": stats["loss"],
                        },
                        step=step,
                    )
            else:
                for batch in (robot_batch, human_batch):
                    stats = train_step_single(model, optimizer, scaler, batch, device, use_amp, loss_cfg)
                    step += 1
                    if stats["embodiment"] == "robot":
                        running["robot_pose"] += stats["pose_loss"]
                        running["robot_joint"] += stats["joint_loss"]
                        running["n_r"] += 1
                        running["n_joint"] += 1
                    else:
                        running["human_pose"] += stats["pose_loss"]
                        running["n_h"] += 1
                    if wandb_run is not None:
                        prefix = stats["embodiment"]
                        log = {
                            f"train/{prefix}_pose_loss": stats["pose_loss"],
                            "train/kl_loss": stats["kld"],
                            "train/total_loss": stats["loss"],
                        }
                        if prefix == "robot":
                            log["train/robot_joint_loss"] = stats["joint_loss"]
                        wandb.log(log, step=step)

            loop.set_postfix(
                rp=running["robot_pose"] / max(1, running["n_r"]),
                rj=running["robot_joint"] / max(1, running["n_joint"]),
                hp=running["human_pose"] / max(1, running["n_h"]),
            )

        avg_rp = running["robot_pose"] / max(1, running["n_r"])
        avg_rj = running["robot_joint"] / max(1, running["n_joint"])
        avg_hp = running["human_pose"] / max(1, running["n_h"])
        avg = (avg_rp + avg_rj + avg_hp) / 3.0
        print(
            f"Epoch {epoch+1}: robot_pose={avg_rp:.6f} robot_joint={avg_rj:.6f} "
            f"human_pose={avg_hp:.6f} avg={avg:.6f}"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_robot_pose": avg_rp,
                    "train/epoch_robot_joint": avg_rj,
                    "train/epoch_human_pose": avg_hp,
                    "train/epoch_avg": avg,
                },
                step=step,
            )

        latest = output_dir / "combined_act_latest.pth"
        torch.save(model.state_dict(), latest)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), output_dir / "combined_act_best.pth")
            print(f"Saved new best -> {output_dir / 'combined_act_best.pth'}")
        if cli.save_every_epochs > 0 and (epoch + 1) % cli.save_every_epochs == 0:
            torch.save(model.state_dict(), output_dir / f"combined_act_epoch_{epoch+1}.pth")

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
