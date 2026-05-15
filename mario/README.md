# Mario World Model

This directory contains a Gymnasium-facing Super Mario Bros world-model pipeline
built on top of `gym-super-mario-bros` and `nes-py`. The current experiment is
level-1-only (`SuperMarioBros-1-1-v0`) and focuses on learning that Mario must
hold the B/run button before jumping to clear larger obstacles.

## Setup

Install the emulator stack in the existing `uv` environment:

```bash
uv pip install gymnasium gym-super-mario-bros==7.4.0 nes-py shimmy
```

The project uses a local Gymnasium-style wrapper in `mario/mariowm/env.py`.
`gym-super-mario-bros` still depends on older Gym internally, so
`mario/mariowm/compat.py` patches NumPy 2.x / `nes-py` overflow issues and the
wrapper exposes modern `reset -> (obs, info)` and
`step -> (obs, reward, terminated, truncated, info)` calls.

## Environment

- Default env: `SuperMarioBros-1-1-v0`.
- Movement set: local `simple_down`, based on `SIMPLE_MOVEMENT` plus `down`,
  `left+down`, and `right+down` combinations.
- `action_idx` stores the executable Joypad action id, while `action` stores an
  extensible multi-hot button vector over `button_names`.
- Current `button_names`: `left`, `right`, `down`, `A`, `B`.
- Current executable actions: `NOOP`, `right`, `right+A`, `right+B`,
  `right+A+B`, `A`, `left`, `down`, `left+down`, `right+down`.
- Frames are resized RGB pixels with shape `84 x 84 x 3`.
- The wrapper truncates episodes after a configurable stall period so failed
  policies do not run indefinitely.

The ROM in `mario/assets/Super Mario Bros. (World).nes` is intentionally ignored
by git; use only legally obtained ROMs.

## Data Policy

The key data-collection fix is `AccelJumpController` in
`mario/mariowm/controllers.py`.

Earlier data over-sampled short, poorly timed jumps, so the model learned to
stall near the first obstacle around x=315. The new policy demonstrates the
correct causal pattern:

```text
hold [right, B] to accelerate
near obstacle windows, hold [right, A, B] for 12-30 frames
resume [right, B] after the jump
occasionally sample [left], [down], and other random legal button-combinations
```

The obstacle windows are jittered per episode so the model sees varied timings.
The mixed data policy is now mostly `accel_jump`, with a smaller amount of
right-biased and random behavior for coverage.

## Dataset Generation

Generate the current level-1-only dataset:

```bash
uv run python mario/generate_mario_dataset.py \
  --output mario/data/mario.h5 \
  --episodes 100 \
  --steps 900 \
  --controller mix \
  --epsilon 0.08 \
  --movement simple_down \
  --env-ids SuperMarioBros-1-1-v0 \
  --visualize-dir mario/visualizations \
  --visualize-episodes 6 \
  --seed 1200
```

Latest generated dataset summary:

```text
transitions: 60,468
episodes: 150
mean_max_x: 514.2
max_x: 899
mean_len: 403.1
run_or_run_jump_action_fraction: 0.726
left_action_fraction: 0.069
down_action_fraction: 0.097
```

HDF5 columns include:

- `pixels`, `next_pixels`: resized RGB frames.
- `action`: behavior action encoded as a multi-hot button vector.
- `action_idx`: executable Joypad behavior action id.
- `expert_action`: `AccelJumpController` expert action as a multi-hot button vector.
- `expert_action_idx`: executable expert action id.
- `reward`: custom transition reward.
- `done`, `terminated`, `truncated`.
- `x_pos`, `next_x_pos`, `delta_x`.
- `coin_delta`, `time_used`, `episode_objective`.
- `score`, `coins`, `time`, `world`, `stage`, `level_id`.
- `episode_idx`, `step_idx`, `ep_len`, `ep_offset`.

HDF5 attributes include `button_names`, `action_names`, and
`action_encoding=button_multihot_v1`.

Render dataset examples:

```bash
uv run python mario/visualize_mario_dataset.py \
  --data mario/data/mario.h5 \
  --output-dir mario/visualizations \
  --count 5 \
  --max-frames 650
```

Example generated videos:

- `mario/visualizations/mario_data_furthest_ep022.mp4`
- `mario/visualizations/mario_data_longest_ep049.mp4`
- `mario/visualizations/mario_data_sample_ep000.mp4`

## Reward / Objective

The saved transition reward is custom, not the raw gym reward:

```text
reward_t = clipped_delta_x
         + coin_reward * coin_delta
         - step_penalty
         + flag_reward if level completed
```

Current defaults:

```text
coin_reward = 25.0
step_penalty = 0.02
flag_reward = 250.0
```

The dataset also stores an episode-level objective:

```text
episode_objective = max_x
                  + coin_objective * final_coins
                  + time_objective * final_time_remaining
                  + flag_objective if level completed
```

Current defaults:

```text
coin_objective = 25.0
time_objective = 0.5
flag_objective = 1000.0
```

The training script currently uses transition reward, `delta_x`, and expert
action labels. `episode_objective` is stored for future value/return training.

## Model

`MarioLeWM` in `mario/mariowm/model.py` is a LeWM-style hybrid pixel latent
model.

Inputs:

- `frame_stack=4`, so the encoder receives 4 recent RGB frames concatenated on
  the channel dimension: `(B, 12, 84, 84)`.
- Multi-hot action vector over the dataset `button_names` attribute.

Encoder:

```text
Conv2d(3 * frame_stack, 32, kernel=8, stride=4, padding=2)
GELU
Conv2d(32, 64, kernel=4, stride=2, padding=1)
GELU
Conv2d(64, 128, kernel=3, stride=2, padding=1)
GELU
Conv2d(128, 128, kernel=3, stride=2, padding=1)
GELU
AdaptiveAvgPool2d(1)
Flatten
Linear(128, 768)
BatchNorm1d(768)
GELU
Linear(768, latent_dim)
```

Default `latent_dim=256`.

Action encoder:

```text
Linear(action_dim, latent_dim)
SiLU
Linear(latent_dim, latent_dim)
```

Here `action_dim = len(button_names)` for model conditioning. The auxiliary
policy head uses a separate `policy_dim = len(action_names)` so control still
selects legal emulator actions.

Predictor:

```text
concat(z_t, action_embedding)
LayerNorm
Linear(2 * latent_dim, 768)
GELU
Dropout(0.1)
Linear(768, 768)
GELU
Dropout(0.1)
Linear(768, latent_dim)
```

Auxiliary heads:

- `policy_head(z_t)`: predicts the executable `AccelJumpController` expert
  `action_idx`.
- `progress_head(z_hat_{t+1})`: predicts normalized clipped `delta_x`.
- `reward_head(z_hat_{t+1})`: predicts normalized custom reward.

## Losses

For each transition:

```text
z_t = encoder(frame_stack_t)
z_{t+1} = encoder(frame_stack_{t+1})
z_hat_{t+1} = predictor(z_t, action_t)
```

Core LeWM-style loss:

```text
L_pred = MSE(z_hat_{t+1}, z_{t+1})
L_sigreg = SIGReg([z_t, z_{t+1}])
```

Auxiliary losses:

```text
L_policy = weighted CE(policy_head(z_t), expert_action_idx)
L_progress = MSE(progress_head(z_hat_{t+1}), delta_x / 20)
L_reward = MSE(reward_head(z_hat_{t+1}), reward / 20)
```

The policy CE is class-weighted by inverse expert-action frequency. This is
important because run+jump is rarer than right+B but crucial for clearing the
first obstacle.

Full objective:

```text
L = L_pred
  + sigreg_weight * L_sigreg
  + policy_weight * L_policy
  + progress_weight * L_progress
  + reward_weight * L_reward
```

## Training

Main longer training run:

```bash
uv run python mario/train_mario_lewm.py \
  --data mario/data/mario.h5 \
  --output mario/outputs/mario_lewm.pt \
  --epochs 18 \
  --batch-size 384 \
  --lr 0.00035 \
  --latent-dim 256 \
  --frame-stack 4 \
  --sigreg-weight 0.035 \
  --sigreg-proj 256 \
  --policy-weight 1.5 \
  --progress-weight 1.0 \
  --reward-weight 0.25 \
  --device auto \
  --control-eval-episodes 3 \
  --control-eval-steps 800 \
  --control-eval-horizon 3 \
  --control-eval-mode planner
```

Weighted fine-tune with best-checkpoint saving:

```bash
uv run python mario/train_mario_lewm.py \
  --data mario/data/mario.h5 \
  --resume mario/outputs/mario_lewm.pt \
  --output mario/outputs/mario_lewm_final.pt \
  --best-output mario/outputs/mario_lewm_best.pt \
  --epochs 8 \
  --batch-size 384 \
  --lr 0.00012 \
  --latent-dim 256 \
  --frame-stack 4 \
  --sigreg-weight 0.035 \
  --sigreg-proj 256 \
  --policy-weight 3.0 \
  --progress-weight 1.0 \
  --reward-weight 0.25 \
  --device auto \
  --control-eval-episodes 5 \
  --control-eval-steps 800 \
  --control-eval-horizon 3 \
  --control-eval-mode planner \
  --metrics-csv mario/outputs/mario_lewm_metrics_best.csv
```

The trainer uses MPS on Apple Silicon when available. For this dataset size, the
fast RAM cache reads the HDF5 arrays sequentially first, then performs NumPy
indexing. This avoids slow single-core `h5py` fancy indexing on compressed HDF5
chunks.

Plot curves:

```bash
uv run python mario/plot_mario_training.py \
  --metrics mario/outputs/mario_lewm_metrics_best.csv \
  --output mario/visualizations/mario_training_curves_best.png
```

## Control

Learned rollout:

```bash
uv run python mario/run_mario_controller.py \
  --controller lewm \
  --model mario/outputs/mario_lewm_best.pt \
  --lewm-mode planner \
  --horizon 3 \
  --steps 900 \
  --seed 5001 \
  --video mario/visualizations/mario_lewm_best_planner_seed5001.mp4 \
  --actions-out mario/outputs/mario_lewm_best_planner_seed5001_actions.txt
```

Acceleration baseline:

```bash
uv run python mario/run_mario_controller.py \
  --controller accel_jump \
  --steps 900 \
  --seed 5001 \
  --video mario/visualizations/mario_accel_jump_seed5001.mp4 \
  --actions-out mario/outputs/mario_accel_jump_seed5001_actions.txt
```

Current best checkpoint result on rendered seed `5001`:

```text
learned LeWM planner: max_x=722, reward=656.0, steps=519
```

The earlier failure mode was a hard plateau around `x ~= 315`; the obstacle-aware
dataset, frame stacking, inverse-frequency action weighting, and best-checkpoint
saving produced checkpoints that clear that bottleneck on some seeds.

## Prediction Visualizations

Teacher-forced diagnostic:

```bash
uv run python mario/visualize_mario_predictions.py \
  --data mario/data/mario.h5 \
  --model mario/outputs/mario_lewm_best.pt \
  --output mario/visualizations/mario_best_prediction_teacher_forced.mp4 \
  --episode 22 \
  --horizon 250 \
  --teacher-forcing
```

Open-loop diagnostic:

```bash
uv run python mario/visualize_mario_predictions.py \
  --data mario/data/mario.h5 \
  --model mario/outputs/mario_lewm_best.pt \
  --output mario/visualizations/mario_best_prediction_open_loop.mp4 \
  --episode 22 \
  --horizon 120
```

Teacher-forced predictions test one-step dynamics. Open-loop predictions drift,
which is expected for LeWM-style models and why control replans frequently.

## Caveats

- This is still a LeWM-style hybrid: latent prediction + SIGReg is the core loss,
  but policy/progress/reward heads are used for Mario control.
- The learned controller does not yet complete level 1-1 reliably.
- The planner horizon is short and operates over primitive actions; macro-actions
  would likely improve long jumps further.
- `gym-super-mario-bros` is an older Gym stack. The local wrapper exposes a
  Gymnasium-style API, but Gym warning messages still appear from the dependency.
