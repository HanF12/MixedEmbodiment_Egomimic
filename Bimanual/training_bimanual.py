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
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Bimanual.core import build, kl_divergence
from Bimanual.config import (
    DEFAULT_NUM_QUERIES,
    MODEL_CAMERA_NAMES,
    STATE_DIM,
    build_run_metadata,
    default_run_name,
    save_run_metadata,
)
from Bimanual.data_synchronization import synchronize_bimanual_with_front
from Bimanual.dataloader_4cam import (
    PreloadedBimanualEpisodeDataset,
    batch_sanity_check,
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bimanual ACT training script")
    parser.add_argument("--data_root", type=str, required=True, help="Path like recording/sessions/teleop_bimanual/0714")
    parser.add_argument("--sync_dir", type=str, default=None, help="Where to write synchronized index CSVs.")
    parser.add_argument("--normalization_path", type=str, default=None, help="Path to save qpos mean/std (.npz).")
    parser.add_argument("--output_dir", type=str, default=None, help="Root directory under which a timestamped run folder will be created.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run folder name (default: current timestamp).")
    parser.add_argument("-e", "--epochs", type=int, default=500)
    parser.add_argument("-b", "--batch", type=int, default=6)
    parser.add_argument("-q", "--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("-g", "--gpu_number", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_skew_s", type=float, default=0.050)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--save_every_epochs", type=int, default=10000, help="Save (and wandb-upload, if enabled) a checkpoint every N epochs.")
    parser.add_argument(
        "--save_periodic_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, also save timestamped epoch snapshots every --save_every epochs.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="bimanual")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", help="wandb mode: online|offline|disabled.")
    parser.add_argument("--dry_run", action="store_true", help="Build sync CSVs, load one batch, and run one forward/backward pass.")
    return parser.parse_args()


def build_sync_csvs(data_root: Path, sync_dir: Path, max_skew_s: float) -> None:
    bird_time_dir = data_root / "bird-realsense-data" / "npy"
    front_time_dir = data_root / "front-realsense-data" / "npy"
    left_cam_time_dir = data_root / "aloha-data" / "left" / "npy"
    right_cam_time_dir = data_root / "aloha-data" / "right" / "npy"
    left_joint_time_dir = data_root / "joint-data" / "left" / "time"
    right_joint_time_dir = data_root / "joint-data" / "right" / "time"

    bird_by_id = {demo_id_from_hash_filename(path): path for path in sorted(bird_time_dir.glob("*.npy"))}
    front_by_id = {demo_id_from_hash_filename(path): path for path in sorted(front_time_dir.glob("*.npy"))}
    left_cam_by_id = {demo_id_from_hash_filename(path): path for path in sorted(left_cam_time_dir.glob("*.npy"))}
    right_cam_by_id = {demo_id_from_hash_filename(path): path for path in sorted(right_cam_time_dir.glob("*.npy"))}
    left_joint_by_id = {demo_id_from_joint_npy(path, prefix="joint_timestamp_"): path for path in sorted(left_joint_time_dir.glob("*.npy"))}
    right_joint_by_id = {demo_id_from_joint_npy(path, prefix="joint_timestamp_"): path for path in sorted(right_joint_time_dir.glob("*.npy"))}

    shared_ids = sorted(
        set(bird_by_id)
        & set(front_by_id)
        & set(left_cam_by_id)
        & set(right_cam_by_id)
        & set(left_joint_by_id)
        & set(right_joint_by_id)
    )
    if not shared_ids:
        raise FileNotFoundError(
            f"No demos with all required streams under {data_root}. "
            "Need left/right wrist, bird, front, and left/right joint timestamps."
        )

    sync_dir.mkdir(parents=True, exist_ok=True)
    for demo_id in shared_ids:
        out_csv = sync_dir / f"{demo_id}.csv"
        synchronize_bimanual_with_front(
            np.load(left_joint_by_id[demo_id]),
            np.load(right_joint_by_id[demo_id]),
            np.load(left_cam_by_id[demo_id]),
            np.load(right_cam_by_id[demo_id]),
            np.load(bird_by_id[demo_id]),
            np.load(front_by_id[demo_id]),
            out_csv,
            max_skew_s=max_skew_s,
            debug=False,
        )


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
        self.state_dim = STATE_DIM


def main() -> None:
    cli = parse_args()
    data_root = Path(cli.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    default_sync_dir = Path(__file__).resolve().parent / "m-synced-csvs" / f"{data_root.name}_bimanual_front"
    sync_dir = Path(cli.sync_dir).expanduser().resolve() if cli.sync_dir else default_sync_dir
    weights_root = Path(cli.output_dir).expanduser().resolve() if cli.output_dir else (Path(__file__).resolve().parent / "weights")
    run_name = cli.run_name or default_run_name()
    output_dir = weights_root / run_name
    normalization_path = (
        Path(cli.normalization_path).expanduser().resolve()
        if cli.normalization_path
        else (output_dir / "normalization_stats_bimanual.npz")
    )

    build_sync_csvs(data_root, sync_dir, cli.max_skew_s)

    wandb_run = None
    if cli.wandb:
        if wandb is None:
            raise RuntimeError(
                "wandb logging requested but wandb is not installed. "
                "Install with `pip install wandb` or disable with no `--wandb`."
            )
        wandb_run = wandb.init(
            project=cli.wandb_project,
            entity=cli.wandb_entity,
            name=cli.wandb_run_name,
            mode=cli.wandb_mode,
            config={
                "epochs": cli.epochs,
                "batch": cli.batch,
                "num_queries": cli.num_queries,
                "gpu_number": cli.gpu_number,
                "data_root": str(data_root),
                "sync_dir": str(sync_dir),
                "lr": cli.lr,
                "weight_decay": cli.weight_decay,
                "run_name": run_name,
            },
        )

    dataset = PreloadedBimanualEpisodeDataset(
        bird_vids_dir=data_root / "bird-realsense-data" / "mp4",
        front_vids_dir=data_root / "front-realsense-data" / "mp4",
        left_arm_vids_dir=data_root / "aloha-data" / "left" / "mp4",
        right_arm_vids_dir=data_root / "aloha-data" / "right" / "mp4",
        left_joint_data_dir=data_root / "joint-data" / "left" / "position",
        right_joint_data_dir=data_root / "joint-data" / "right" / "position",
        sync_csv_dir=sync_dir,
        num_queries=cli.num_queries,
        max_demos=cli.max_demos,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(normalization_path, qpos_mean=dataset.joint_mean.numpy(), qpos_std=dataset.joint_std.numpy())
    print(f"Normalization statistics saved to {normalization_path}")
    if wandb_run is not None:
        wandb.save(str(normalization_path.resolve()))
    metadata_path = save_run_metadata(
        output_dir,
        build_run_metadata(
            data_root=data_root,
            sync_dir=sync_dir,
            num_queries=cli.num_queries,
            max_skew_s=cli.max_skew_s,
        ),
    )
    print(f"Run metadata saved to {metadata_path}")

    n = len(dataset)
    if n == 0:
        raise RuntimeError("Dataset is empty after sync.")
    n_items = len(dataset)
    num_demos = int(getattr(dataset, "num_demos", 0) or 0)
    num_samples = int(getattr(dataset, "num_samples", 0) or 0)
    num_views = int(getattr(dataset, "num_views", 1) or 1)
    print(f"Dataset items: {n_items} (samples={num_samples}, demos={num_demos}, views={num_views})")

    if num_demos > 1:
        rng = np.random.default_rng(42)
        demo_ids = np.arange(num_demos)
        rng.shuffle(demo_ids)

        train_demo_len = max(1, int(0.95 * num_demos))
        if train_demo_len >= num_demos:
            train_demo_len = num_demos - 1

        train_indices = list(map(int, demo_ids[:train_demo_len]))
        val_indices = list(map(int, demo_ids[train_demo_len:]))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        val_len = len(val_dataset)
        print(
            f"Demo split: {len(train_indices)} train demos / {len(val_indices)} val demos "
            f"-> {len(train_dataset)} train items / {len(val_dataset)} val items"
        )
    else:
        print("WARNING: insufficient demos for demo-level split; falling back to sample-level split.")
        train_len = max(1, int(0.95 * n_items)) if n_items > 1 else n_items
        val_len = n_items - train_len
        if val_len == 0 and n_items > 1:
            train_len = n_items - 1
            val_len = 1
        print(f"Sample split: {n_items} items ({train_len} train, {val_len} val)")
        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

    loader_kwargs = dict(
        batch_size=cli.batch,
        num_workers=cli.num_workers,
        pin_memory=True,
        persistent_workers=cli.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_shuffle = len(val_dataset) > 0
    val_loader = DataLoader(val_dataset, shuffle=val_shuffle, **loader_kwargs)
    batch_sanity_check(train_loader)

    device = torch.device(f"cuda:{cli.gpu_number}" if torch.cuda.is_available() else "cpu")
    model = build(Args(cli.num_queries)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    batch_counter = 0

    if cli.dry_run:
        qpos, image, actions, is_pad = next(iter(train_loader))
        qpos = qpos.to(device)
        image = image.to(device)
        actions = actions.to(device)
        is_pad = is_pad.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            output, _, (mu, logvar) = model(qpos, image, env_state=None, actions=actions, is_pad=is_pad)
            total_kld, *_ = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(output[..., :14], actions, reduction="none")
            mask = (~is_pad).unsqueeze(-1)
            rec_loss = (all_l1 * mask).sum() / mask.sum()
            loss = rec_loss + total_kld[0] * 10
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Dry run passed. loss={float(loss):.6f}")
        return

    if cli.epochs <= 0:
        raise ValueError("--epochs must be > 0 unless --dry_run is used")

    best_val = float("inf")
    for epoch in range(cli.epochs):
        model.train()
        running_rec_loss = 0.0
        running_total_loss = 0.0
        running_kld = 0.0
        train_batches = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cli.epochs}", unit="batch")
        for qpos, image, actions, is_pad in loop:
            qpos = qpos.to(device)
            image = image.to(device)
            actions = actions.to(device)
            is_pad = is_pad.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                output, _, (mu, logvar) = model(qpos, image, env_state=None, actions=actions, is_pad=is_pad)
                total_kld, *_ = kl_divergence(mu, logvar)
                all_l1 = F.l1_loss(output[..., :14], actions, reduction="none")
                mask = (~is_pad).unsqueeze(-1)
                rec_loss = (all_l1 * mask).sum() / mask.sum()
                kld = total_kld[0]
                loss = rec_loss + kld * 10
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            rec_loss_item = float(rec_loss.detach().cpu().item())
            kld_item = float(kld.detach().cpu().item())
            total_loss_item = float(loss.detach().cpu().item())

            running_rec_loss += rec_loss_item
            running_kld += kld_item
            running_total_loss += total_loss_item
            train_batches += 1
            batch_counter += 1
            loop.set_postfix(
                rec=running_rec_loss / train_batches,
                kld=running_kld / train_batches,
                loss=running_total_loss / train_batches,
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "train/rec_loss_step": rec_loss_item,
                        "train/kld_step": kld_item,
                        "train/loss_step": total_loss_item,
                        "lr": optimizer.param_groups[0].get("lr", None),
                    },
                    step=batch_counter,
                )

        model.eval()
        running_val = 0.0
        val_batches = 0
        with torch.no_grad():
            for qpos, image, actions, is_pad in val_loader:
                qpos = qpos.to(device)
                image = image.to(device)
                actions = actions.to(device)
                is_pad = is_pad.to(device)
                with autocast(enabled=device.type == "cuda"):
                    pred, _, _ = model(qpos, image, None)
                    all_l1 = F.l1_loss(pred[..., :14], actions, reduction="none")
                    mask = (~is_pad).unsqueeze(-1)
                    rec_val = (all_l1 * mask).sum() / mask.sum()
                running_val += rec_val.item()
                val_batches += 1

        avg_rec = running_rec_loss / train_batches
        avg_kld = running_kld / train_batches
        avg_train = running_total_loss / train_batches
        avg_val = running_val / max(1, val_batches)
        print(f"Epoch {epoch + 1}: train loss={avg_train:.6f} (rec {avg_rec:.6f}, kld {avg_kld:.6f}) val rec={avg_val:.6f}")
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/rec_loss": avg_rec,
                    "train/kld": avg_kld,
                    "train/loss": avg_train,
                    "val/rec_loss": avg_val,
                    "lr": optimizer.param_groups[0].get("lr", None),
                },
                step=batch_counter,
            )

        latest_path = output_dir / "bimanual_act_latest.pth"
        torch.save(model.state_dict(), latest_path)
        if wandb_run is not None:
            wandb.save(str(latest_path.resolve()))
        if avg_val < best_val:
            best_val = avg_val
            best_path = output_dir / "bimanual_act_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best checkpoint to {best_path}")
            if wandb_run is not None:
                wandb.save(str(best_path.resolve()))
        save_every_epochs = int(getattr(cli, "save_every_epochs", 0) or 0)
        if save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            epoch_ckpt_path = output_dir / f"bimanual_act_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f"Checkpoint saved: {epoch_ckpt_path}")
            if wandb_run is not None:
                wandb.save(str(epoch_ckpt_path.resolve()))
        if cli.save_periodic_checkpoints and (epoch + 1) % cli.save_every == 0:
            periodic_path = output_dir / f"bimanual_act_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), periodic_path)
            print(f"Saved periodic checkpoint to {periodic_path}")
            if wandb_run is not None:
                wandb.save(str(periodic_path.resolve()))

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
