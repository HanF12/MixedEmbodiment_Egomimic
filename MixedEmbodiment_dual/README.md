# MixedEmbodiment_dual (Method A bridge)

Copy of `MixedEmbodiment` with a different mixed-data training recipe.

## What changed

1. **Mixed demos are folded into the human-side loader** (concat). There is no separate mixed loader that gets recycled every step just because it is shorter than human.
2. Each batch from that loader is still **all-human or all-mixed** (homogeneous sampler).
3. On a **mixed** batch we run **two routes**:
   - **Robot route:** joints → robot proprio/CVAE, mixed camera mask (bird + active wrist), joint head + pose head
   - **Human route:** pose → human proprio/CVAE, **bird-only** camera mask, pose head (no joints)
4. Loss: `L_mixed = mixed_lambda * 0.5 * (L_robot_route + L_human_route)`, then averaged with the pure-robot batch loss as usual.

## Two pose modes for the human route

| Flag | Behavior |
|---|---|
| `--mixed_human_pose full8` (**default**) | Human route uses all **8** glued dims (hand + robot EEF) |
| `--mixed_human_pose hand4` | Human route keeps only the **hand** half; robot half zeroed and masked in the pose loss |

Both modes use correct masks on each route (mixed cams for robot route, human cams for human route).

## Run

From repo root:

```bash
# full 8D human-route pose (default)
python -m MixedEmbodiment_dual.training_combined --sessions_root sessions -g 0 --wandb --jpeg_in_ram

# hand-only 4D human-route pose
python -m MixedEmbodiment_dual.training_combined --sessions_root sessions -g 0 --wandb --jpeg_in_ram \
  --mixed_human_pose hand4
```

Original `MixedEmbodiment/` is unchanged.
