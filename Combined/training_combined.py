"""
Mixed human + robot ACT training entrypoint.

Does not modify Bimanual/. Uses Combined sync + dataloaders + MixedDETRVAE.
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

from Combined.config import (  # noqa: E402
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_HUMAN,
    EMBODIMENT_ROBOT,
    HUMAN_STATE_DIM,
    MODEL_CAMERA_NAMES,
    ROBOT_STATE_DIM,
    build_run_metadata,
    default_run_name,
    save_run_metadata,
)
from Combined.core import build, kl_divergence  # noqa: E402
from Combined.data_synchronization import (  # noqa: E402
    synchronize_human_hands,
    synchronize_robot_bimanual,
)
from Combined.dataloader_human import HumanEpisodeDataset, collate_homogeneous  # noqa: E402
from Combined.dataloader_robot import RobotEpisodeDataset  # noqa: E402
from Combined.dataloader_utils import (  # noqa: E402
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    demo_id_from_pose_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combined mixed human+robot ACT training")
    p.add_argument("--robot_data_root", type=str, default=None, help="e.g. recording/sessions/teleop_bimanual/0714")
    p.add_argument(
        "--human_data_root",
        type=str,
        default=None,
        help="e.g. recording/sessions/human_hands_bimanual_raw/0714",
    )
    p.add_argument("--robot_sync_dir", type=str, default=None)
    p.add_argument("--human_sync_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("-e", "--epochs", type=int, default=500)
    p.add_argument("-b", "--batch", type=int, default=4)
    p.add_argument("-q", "--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    p.add_argument("-g", "--gpu_number", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_skew_s", type=float, default=0.050)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_demos", type=int, default=None, help="Cap demos per modality")
    p.add_argument("--resize_factor", type=float, default=1.0)
    p.add_argument("--max_sync_rows", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_every_epochs", type=int, default=1000)
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
        help="If robot data is missing, run a synthetic robot batch to exercise the robot head.",
    )
    return p.parse_args()


def apply_smoke_defaults(cli: argparse.Namespace) -> argparse.Namespace:
    if not cli.smoke:
        return cli
    cli.dry_run = True
    cli.cpu = True
    cli.wandb = False
    if cli.max_demos is None:
        cli.max_demos = 1
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
    if cli.robot_data_root is None:
        cli.synthetic_robot = True
    print(
        "SMOKE MODE: "
        f"max_demos={cli.max_demos} batch={cli.batch} workers={cli.num_workers} "
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


def build_robot_sync_csvs(data_root: Path, sync_dir: Path, max_skew_s: float, max_demos: int | None) -> None:
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
    ids = sorted(set(bird) & set(front) & set(left_c) & set(right_c) & set(left_j) & set(right_j))
    if max_demos is not None and max_demos > 0:
        ids = ids[:max_demos]
    if not ids:
        raise FileNotFoundError(f"No complete robot demos under {data_root}")
    sync_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building robot sync for {len(ids)} demos -> {sync_dir}")
    for demo_id in ids:
        synchronize_robot_bimanual(
            np.load(left_j[demo_id]),
            np.load(right_j[demo_id]),
            np.load(left_c[demo_id]),
            np.load(right_c[demo_id]),
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            sync_dir / f"{demo_id}.csv",
            max_skew_s=max_skew_s,
            debug=False,
        )


def build_human_sync_csvs(data_root: Path, sync_dir: Path, max_skew_s: float, max_demos: int | None) -> None:
    bird = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "bird-realsense-data" / "npy").glob("*.npy"))}
    front = {demo_id_from_hash_filename(p): p for p in sorted((data_root / "front-realsense-data" / "npy").glob("*.npy"))}
    pose_dir = data_root / "bird-realsense-data" / "combined_npz"
    pose = {demo_id_from_pose_npz(p): p for p in sorted(pose_dir.glob("*.npz"))}
    ids = sorted(set(bird) & set(front) & set(pose))
    if max_demos is not None and max_demos > 0:
        ids = ids[:max_demos]
    if not ids:
        raise FileNotFoundError(
            f"No complete human demos under {data_root}. "
            "Need bird/front npy timestamps and combined_npz pose NPZs."
        )
    sync_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building human sync for {len(ids)} demos -> {sync_dir}")
    for demo_id in ids:
        pose_npz = np.load(pose[demo_id])
        synchronize_human_hands(
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            pose_npz["timestamps"],
            sync_dir / f"{demo_id}.csv",
            max_skew_s=max_skew_s,
            debug=False,
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
    """Tiny fake robot batch to validate robot head when teleop data is unavailable."""
    device = device or torch.device("cpu")
    return {
        "embodiment": EMBODIMENT_ROBOT,
        "images": torch.randn(batch_size, 4, 3, h, w, device=device),
        "camera_mask": torch.ones(batch_size, 4, device=device),
        "state": torch.randn(batch_size, ROBOT_STATE_DIM, device=device),
        "actions": torch.randn(batch_size, num_queries, ROBOT_STATE_DIM, device=device),
        "is_pad": torch.zeros(batch_size, num_queries, dtype=torch.bool, device=device),
        "action_dim": ROBOT_STATE_DIM,
    }


def train_step(model, optimizer, scaler, batch, device, use_amp: bool) -> dict:
    state = batch["state"].to(device)
    images = batch["images"].to(device)
    actions = batch["actions"].to(device)
    is_pad = batch["is_pad"].to(device)
    camera_mask = batch["camera_mask"].to(device)
    embodiment = int(batch["embodiment"])
    action_dim = int(batch["action_dim"])

    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=use_amp):
        # pred: [B,K,D]
        pred, _, (mu, logvar) = model(
            state,
            images,
            embodiment=embodiment,
            camera_mask=camera_mask,
            actions=actions,
            is_pad=is_pad,
        )
        total_kld, *_ = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(pred[..., :action_dim], actions[..., :action_dim], reduction="none")
        mask = (~is_pad).unsqueeze(-1)
        rec_loss = (all_l1 * mask).sum() / mask.sum().clamp(min=1)
        loss = rec_loss + total_kld[0] * 10

    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach().cpu()),
        "rec": float(rec_loss.detach().cpu()),
        "kld": float(total_kld[0].detach().cpu()),
        "embodiment": "robot" if embodiment == EMBODIMENT_ROBOT else "human",
        "action_dim": action_dim,
        "image_shape": tuple(images.shape),
        "state_shape": tuple(state.shape),
        "pred_shape": tuple(pred.shape),
    }


def main() -> None:
    cli = apply_smoke_defaults(parse_args())
    pkg = Path(__file__).resolve().parent

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

    weights_root = Path(cli.output_dir).expanduser().resolve() if cli.output_dir else (pkg / "weights")
    run_name = cli.run_name or default_run_name()
    if cli.smoke and cli.run_name is None:
        run_name = f"{run_name}_smoke"
    output_dir = weights_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_ds = None
    human_ds = None

    if robot_root is not None:
        if not robot_root.exists():
            raise FileNotFoundError(robot_root)
        build_robot_sync_csvs(robot_root, robot_sync, cli.max_skew_s, cli.max_demos)
        robot_ds = RobotEpisodeDataset(
            bird_vids_dir=robot_root / "bird-realsense-data" / "mp4",
            front_vids_dir=robot_root / "front-realsense-data" / "mp4",
            left_arm_vids_dir=robot_root / "aloha-data" / "left" / "mp4",
            right_arm_vids_dir=robot_root / "aloha-data" / "right" / "mp4",
            left_joint_data_dir=robot_root / "joint-data" / "left" / "position",
            right_joint_data_dir=robot_root / "joint-data" / "right" / "position",
            sync_csv_dir=robot_sync,
            num_queries=cli.num_queries,
            max_demos=cli.max_demos,
            resize_factor=cli.resize_factor,
            max_sync_rows=cli.max_sync_rows,
        )
        np.savez(
            output_dir / "normalization_stats_robot.npz",
            qpos_mean=robot_ds.state_mean.numpy(),
            qpos_std=robot_ds.state_std.numpy(),
        )

    if human_root is not None:
        if not human_root.exists():
            raise FileNotFoundError(human_root)
        build_human_sync_csvs(human_root, human_sync, cli.max_skew_s, cli.max_demos)
        human_ds = HumanEpisodeDataset(
            bird_vids_dir=human_root / "bird-realsense-data" / "mp4",
            front_vids_dir=human_root / "front-realsense-data" / "mp4",
            pose_npz_dir=human_root / "bird-realsense-data" / "combined_npz",
            sync_csv_dir=human_sync,
            num_queries=cli.num_queries,
            max_demos=cli.max_demos,
            resize_factor=cli.resize_factor,
            max_sync_rows=cli.max_sync_rows,
        )
        np.savez(
            output_dir / "normalization_stats_human.npz",
            pose_mean=human_ds.state_mean.numpy(),
            pose_std=human_ds.state_std.numpy(),
        )

    meta = build_run_metadata(
        robot_data_root=robot_root,
        human_data_root=human_root,
        robot_sync_dir=robot_sync if robot_root else None,
        human_sync_dir=human_sync if human_root else None,
        num_queries=cli.num_queries,
        max_skew_s=cli.max_skew_s,
    )
    save_run_metadata(output_dir, meta)

    device = torch.device("cpu" if cli.cpu or not torch.cuda.is_available() else f"cuda:{cli.gpu_number}")
    print(f"Using device: {device}")
    model = build(Args(cli.num_queries)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    robot_loader = make_loader(robot_ds, cli.batch, cli.num_workers, shuffle=True) if robot_ds is not None else None
    human_loader = make_loader(human_ds, cli.batch, cli.num_workers, shuffle=True) if human_ds is not None else None

    if cli.dry_run:
        print("--- Combined dry run ---")
        if human_loader is not None:
            batch = next(iter(human_loader))
            stats = train_step(model, optimizer, scaler, batch, device, use_amp)
            print(f"Human step OK: {stats}")
        if robot_loader is not None:
            batch = next(iter(robot_loader))
            stats = train_step(model, optimizer, scaler, batch, device, use_amp)
            print(f"Robot step OK: {stats}")
        elif cli.synthetic_robot:
            batch = synthetic_robot_batch(cli.batch, cli.num_queries, device=device)
            stats = train_step(model, optimizer, scaler, batch, device, use_amp)
            print(f"Synthetic robot step OK: {stats}")
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
    for epoch in range(cli.epochs):
        model.train()
        robot_iter = iter(robot_loader)
        human_iter = iter(human_loader)
        n_batches = max(len(robot_loader), len(human_loader))
        running = {"robot_rec": 0.0, "human_rec": 0.0, "n_r": 0, "n_h": 0}
        loop = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{cli.epochs}", unit="batch")
        for _ in loop:
            # Alternate modalities each micro-step
            for loader_iter, name in ((robot_iter, "robot"), (human_iter, "human")):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    continue
                stats = train_step(model, optimizer, scaler, batch, device, use_amp)
                step += 1
                if name == "robot":
                    running["robot_rec"] += stats["rec"]
                    running["n_r"] += 1
                else:
                    running["human_rec"] += stats["rec"]
                    running["n_h"] += 1
                if wandb_run is not None:
                    wandb.log({f"train/{name}_rec_step": stats["rec"], f"train/{name}_loss_step": stats["loss"]}, step=step)
            loop.set_postfix(
                r=running["robot_rec"] / max(1, running["n_r"]),
                h=running["human_rec"] / max(1, running["n_h"]),
            )

        avg_r = running["robot_rec"] / max(1, running["n_r"])
        avg_h = running["human_rec"] / max(1, running["n_h"])
        avg = 0.5 * (avg_r + avg_h)
        print(f"Epoch {epoch+1}: robot_rec={avg_r:.6f} human_rec={avg_h:.6f} avg={avg:.6f}")
        if wandb_run is not None:
            wandb.log({"epoch": epoch + 1, "train/robot_rec": avg_r, "train/human_rec": avg_h, "train/avg_rec": avg}, step=step)

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
