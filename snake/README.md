# Snake World Model

This repo now includes a seedable black-and-white wrap-around Snake simulator,
dataset exporter, a compact Snake-specific world-model trainer, prediction
visualizers, and controllers that emit playable action signals and videos.

## Run The Simulator

Keyboard mode:

```bash
uv run python snake/run_snake_controller.py --controller keyboard --seed 7
```

Autonomous MPC mode:

```bash
uv run python snake/run_snake_controller.py \
  --controller mpc \
  --horizon 4 \
  --steps 180 \
  --seed 7 \
  --video snake/visualizations/snake_solution_mpc_wrap.mp4 \
  --actions-out snake/outputs/snake_actions.txt
```

The action signal is written as one action name per line in
`snake/outputs/snake_actions.txt`.

Scripted mode:

```bash
uv run python snake/run_snake_controller.py \
  --controller scripted \
  --actions right,right,down,left \
  --steps 20
```

Learned-model one-step control mode:

```bash
uv run python snake/run_snake_controller.py \
  --controller model \
  --model snake/outputs/snake_world_model.pt \
  --steps 120 \
  --video snake/visualizations/snake_model_controller.mp4
```

LeWM latent controller mode:

```bash
uv run python snake/run_snake_controller.py \
  --controller lewm \
  --lewm-mode planner \
  --lewm-model snake/outputs/snake_lewm.pt \
  --horizon 5 \
  --steps 180 \
  --seed 7 \
  --video snake/visualizations/snake_lewm_planner.mp4 \
  --actions-out snake/outputs/snake_lewm_planner_actions.txt
```

## Generate Dataset

```bash
uv run python snake/generate_snake_dataset.py \
  --output snake/data/snake.h5 \
  --episodes 360 \
  --steps 240 \
  --controller mix \
  --epsilon 0.15 \
  --horizon 2 \
  --seed 100
```

The generator also records `expert_action` labels from the greedy controller.
Those labels are used only for lightweight policy/probe heads on top of the
latent representation; the core LeWM loss remains next-latent prediction plus
SIGReg.

The HDF5 file contains:

- `pixels`: black-and-white rendered frames, compatible with the existing LeWM image path.
- `action`: one-hot action vectors `[up, right, down, left]`.
- `state`: structured Snake channels for body, head, food, and direction.
- `next_state`, `next_pixels`, `reward`, `done`, `score`, `wrapped`, `collision`, `episode_idx`, `step_idx`.
- `ep_len`, `ep_offset`: episode indexing used by `stable_worldmodel.data.HDF5Dataset`.

Render dataset examples:

```bash
uv run python snake/visualize_snake_dataset.py \
  --data snake/data/snake.h5 \
  --output-dir snake/visualizations \
  --count 4
```

## Train Compact Snake World Model

```bash
uv run python snake/train_snake_world_model.py \
  --data snake/data/snake.h5 \
  --output snake/outputs/snake_world_model.pt \
  --epochs 16 \
  --batch-size 512 \
  --device auto
```

This trains a small action-conditioned dynamics model that predicts next Snake
state, reward, and done from `(state, action)`. On this Mac, `--device auto`
selects Apple Metal/MPS. On multi-CUDA machines, the trainer uses all visible
CUDA GPUs through `DataParallel`.

## Train Snake LeWM Latent Model

```bash
uv run python snake/train_snake_lewm.py \
  --data snake/data/snake.h5 \
  --output snake/outputs/snake_lewm.pt \
  --epochs 12 \
  --batch-size 512 \
  --device auto
```

On Apple Silicon, PyTorch exposes the Mac GPU as one logical Metal device
(`mps:0`). The trainer prints the selected backend and preloads selected HDF5
rows into RAM by default so the MPS device is not starved by compressed random
HDF5 reads.

Render LeWM latent prediction checks:

```bash
uv run python snake/visualize_snake_lewm_predictions.py \
  --data snake/data/snake.h5 \
  --model snake/outputs/snake_lewm.pt \
  --output snake/visualizations/snake_lewm_prediction_teacher_forced.mp4 \
  --episode 206 \
  --horizon 120 \
  --teacher-forcing
```

Render prediction checks:

```bash
uv run python snake/visualize_snake_predictions.py \
  --data snake/data/snake.h5 \
  --model snake/outputs/snake_world_model.pt \
  --output snake/visualizations/snake_prediction_teacher_forced.mp4 \
  --episode 206 \
  --horizon 120 \
  --teacher-forcing
```

## Train LeWM On Snake Pixels

The file `snake/config/train/data/snake.yaml` points the existing LeWM training script
at `snake/data/snake.h5`:

```bash
uv run python train.py data=snake wandb.enabled=false trainer.max_epochs=10
```

To keep the root repo close to the upstream LeWM layout, this Snake-specific
Hydra data config lives under `snake/config`. If you want to use the upstream
`train.py data=snake` command, copy or symlink this file into
`config/train/data/snake.yaml` first. The config consumes the same `pixels` and
`action` columns exported by the dataset generator.

## Reproducible Methods

This section documents the Snake experiment end-to-end: simulator dynamics,
dataset generation, action sources, model architecture, losses, optimization,
control, and evaluation. It is intended to make the current results reproducible
and to clarify which parts are canonical LeWM-style training versus practical
Snake-specific scaffolding.

### Environment

The environment is implemented in `snake/snakewm/game.py` as `SnakeGame`.

Board and rendering:

- The default board is `12 x 12` cells.
- The rendered observation is a compact retro black-and-white RGB image.
- Default render size is `100 x 100 x 3`, produced from `cell_size=8` and `border=2`.
- The game is seedable via `SnakeGame(seed=...)` and deterministic for a fixed seed and action sequence.

State:

- The snake is stored as an ordered list of `(x, y)` cells, with the head first.
- The default initial length is `3`.
- Food is placed uniformly at random in an unoccupied cell.
- Direction is one of `[up, right, down, left]`.

Actions:

- The action space is discrete with four actions:
  `0=up`, `1=right`, `2=down`, `3=left`.
- Actions are exported as one-hot vectors `[up, right, down, left]`.
- Direct reversal is disallowed when the snake length is greater than one. If the selected action is the opposite of the current direction, the simulator keeps the current direction.

Dynamics:

- The board is toroidal. Leaving one edge wraps to the opposite edge.
- Example: moving right from `x = width - 1` re-enters at `x = 0`.
- Eating food increases score by `1`, grows the snake by one cell, and respawns food.
- Moving without eating removes the tail.
- Self-collision terminates the episode.
- A timeout terminates the episode after `width * height * 2` steps without food.

Rewards and terminal values are recorded for diagnostics and baseline models, but
they are not required by the core LeWM latent prediction objective.

### Structured State Channels

The simulator also exposes a privileged structured state used for probes,
diagnostics, and the older supervised dynamics baseline. It is not the primary
input to the Snake LeWM pixel model.

`state` has shape `(height, width, 7)`:

- Channel `0`: snake body excluding the head.
- Channel `1`: snake head.
- Channel `2`: food.
- Channels `3:7`: one-hot direction planes for `[up, right, down, left]`.

The LeWM model consumes `pixels`; the structured state is used to evaluate what
the latent representation has learned and to train auxiliary probe heads.

### Dataset Generation

Datasets are generated with `generate_snake_dataset.py` and saved as HDF5.

The most recent dataset was generated with:

```bash
uv run python snake/generate_snake_dataset.py \
  --output snake/data/snake.h5 \
  --episodes 420 \
  --steps 260 \
  --controller mix \
  --epsilon 0.12 \
  --horizon 2 \
  --seed 500
```

This produced:

```text
transitions: 84,355
episodes: 420
wrap_events: 33,586
collisions: 194
mean_final_score: 6.198
```

Each row corresponds to one transition `(o_t, a_t, o_{t+1})`.

HDF5 columns:

- `pixels`: current rendered RGB frame, shape `(N, 100, 100, 3)`, `uint8`.
- `next_pixels`: next rendered RGB frame.
- `state`: current privileged state, shape `(N, 12, 12, 7)`.
- `next_state`: next privileged state.
- `action`: behavior action as one-hot vector, shape `(N, 4)`.
- `action_idx`: behavior action as integer.
- `expert_action`: greedy expert action as one-hot vector.
- `expert_action_idx`: greedy expert action as integer.
- `reward`: simulator reward for the transition.
- `done`: terminal flag.
- `score`: score after the transition.
- `wrapped`: whether the transition crossed a board edge.
- `edge_crossings`: cumulative edge crossings in the episode.
- `collision`: whether the transition ended in self-collision.
- `controller_id`: behavior controller id.
- `episode_idx`: episode id for each row.
- `step_idx`: step number within the episode.
- `ep_len`: episode lengths.
- `ep_offset`: global row offset of each episode.

### Behavior Policies And Action Generation

The dataset deliberately mixes good, bad, and edge-case behavior. This matters
because world models need dynamics coverage, not only expert demonstrations.

Controllers are implemented in `snake/snakewm/controllers.py`.

Random controller:

- Samples uniformly from legal actions.
- Produces failures, collisions, and diverse off-policy states.

Greedy controller:

- Uses breadth-first search to find a safe shortest path to food.
- If no path is found, it chooses the action with the largest reachable free-space area.
- This controller also provides `expert_action` labels for the auxiliary policy head.

Edge-crossing controller:

- Biases motion toward board edges and wrap-around transitions.
- Ensures the dataset contains many toroidal boundary crossings.

Exact simulator MPC controller:

- Enumerates short action sequences using the real simulator as the dynamics model.
- Scores rollouts by food acquisition, reachable area, and toroidal distance to food.
- Used as a strong non-learned baseline and as one source of behavior data.

Mixed controller:

- Used for the main dataset.
- Cycles through `greedy`, `random`, `edge`, `mpc`, `random`, `edge` by episode.
- Adds epsilon-random legal action injection with `epsilon=0.12` in the latest dataset.

This means the learned model sees successful food collection, random wandering,
edge wrapping, self-collisions, and non-expert behavior.

### Models

There are two learned model families in this repo.

The main model is `SnakeLeWM` in `snake/snakewm/model.py`. It is the LeWM-style pixel
latent model used for current control results.

The older baseline is `SnakeDynamicsModel`, a supervised structured-state model
that predicts `next_state`, `reward`, and `done`. It is useful for comparison,
but it is not the main LeWM-style result.

### SnakeLeWM Architecture

`SnakeLeWM` maps pixels to a compact latent vector and predicts the next latent
conditioned on an action.

Inputs:

- `pixels`: RGB image tensor, shape `(B, 3, 100, 100)`.
- `action`: one-hot vector, shape `(B, 4)`.

Pixel preprocessing:

- If pixel values are greater than `2`, they are divided by `255.0`.
- No ImageNet normalization is used in this Snake-specific trainer.

Encoder backbone:

```text
Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
GELU
Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
GELU
Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
GELU
Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
GELU
AdaptiveAvgPool2d(1)
Flatten
```

Projection head:

```text
Linear(128, 512)
BatchNorm1d(512)
GELU
Linear(512, 192)
```

The default latent dimension is `192`, matching the LeWM paper's compact latent
size.

Action encoder:

```text
Linear(4, 192)
SiLU
Linear(192, 192)
```

Latent predictor:

```text
concat(z_t, action_embedding) -> 384 dims
LayerNorm(384)
Linear(384, 512)
GELU
Dropout(0.1)
Linear(512, 512)
GELU
Dropout(0.1)
Linear(512, 192)
```

The predictor outputs `z_hat_{t+1}`.

Auxiliary heads:

- `policy_head`: predicts greedy expert action from `z_t`.
- `body_head`: predicts body occupancy from a latent.
- `head_head`: predicts head cell from a latent.
- `food_head`: predicts food cell from a latent.
- `direction_head`: predicts direction from a latent.

These heads are practical scaffolding for diagnostics and control. The core
world-model signal remains latent prediction plus SIGReg.

### Core LeWM Loss

For each transition, the model computes:

```text
z_t = encoder(pixels_t)
z_{t+1} = encoder(pixels_{t+1})
z_hat_{t+1} = predictor(z_t, action_t)
```

The latent prediction loss is:

```text
L_pred = MSE(z_hat_{t+1}, z_{t+1})
```

SIGReg is applied to the stack `[z_t, z_{t+1}]`:

```text
L_sigreg = SIGReg(stack([z_t, z_{t+1}]))
```

SIGReg encourages the latent distribution to match an isotropic Gaussian using
random one-dimensional projections and the Epps-Pulley normality statistic. This
prevents representation collapse, where the encoder maps all observations to the
same vector.

The core LeWM loss is:

```text
L_core = L_pred + lambda_sigreg * L_sigreg
```

Current default:

```text
lambda_sigreg = 0.05
num_sigreg_projections = 256
sigreg_knots = 17
```

The original LeWM paper commonly uses `1024` projections and `lambda` around
`0.1`; this Snake trainer uses fewer projections for faster local MacBook runs.

### Auxiliary Probe And Policy Losses

The Snake model includes auxiliary losses to make the latent space directly
measurable and to produce a usable controller.

Current-state probe loss:

```text
L_probe_current = body_bce(z_t) + head_ce(z_t) + food_ce(z_t) + direction_ce(z_t)
```

Predicted-next-state probe loss:

```text
L_probe_next = body_bce(z_hat_{t+1}) + head_ce(z_hat_{t+1}) + food_ce(z_hat_{t+1}) + direction_ce(z_hat_{t+1})
```

Body uses binary cross entropy with a positive-class weight computed from body
cell sparsity. Head, food, and direction use cross entropy.

Policy loss:

```text
L_policy = CE(policy_head(z_t), expert_action_idx)
```

The expert action is generated by the greedy BFS controller during dataset
generation.

Full training loss:

```text
L_total = L_pred
        + lambda_sigreg * L_sigreg
        + probe_weight * (L_probe_current + L_probe_next)
        + policy_weight * L_policy
```

Main training run used:

```text
lambda_sigreg = 0.05
probe_weight = 0.15
policy_weight = 0.8
```

Important interpretation:

- The model is LeWM-style, not pure canonical LeWM.
- The latent dynamics are trained with LeWM's prediction plus SIGReg objective.
- Probe and policy heads are added to make the representation controllable and measurable in this small game.

### Optimization

Training is implemented in `train_snake_lewm.py`.

Optimizer:

```text
AdamW
learning_rate = 5e-4 for initial training
learning_rate = 2.5e-4 for continued training
weight_decay = 1e-4
gradient_clip_norm = 1.0
```

Batching:

```text
batch_size = 512
max_samples = 50,000 for the main training run
validation_split = 0.1
```

Hardware:

- `--device auto` selects `mps` on Apple Silicon when available.
- PyTorch exposes Apple Silicon GPU as one logical Metal device: `mps:0`.
- The trainer prints backend information and runs an MPS smoke test.
- Selected HDF5 rows are preloaded into RAM by default to avoid starving the GPU with compressed random HDF5 reads.

Initial training command:

```bash
uv run python snake/train_snake_lewm.py \
  --data snake/data/snake.h5 \
  --output snake/outputs/snake_lewm.pt \
  --epochs 12 \
  --batch-size 512 \
  --lr 0.0005 \
  --sigreg-weight 0.05 \
  --sigreg-proj 256 \
  --probe-weight 0.15 \
  --policy-weight 0.8 \
  --max-samples 50000 \
  --device auto \
  --num-workers 0 \
  --log-every 20
```

Continued training command:

```bash
uv run python snake/train_snake_lewm.py \
  --data snake/data/snake.h5 \
  --resume snake/outputs/snake_lewm.pt \
  --output snake/outputs/snake_lewm_continued.pt \
  --epochs 8 \
  --batch-size 512 \
  --lr 0.00025 \
  --sigreg-weight 0.05 \
  --sigreg-proj 256 \
  --probe-weight 0.15 \
  --policy-weight 0.8 \
  --max-samples 50000 \
  --device auto \
  --num-workers 0 \
  --log-every 20 \
  --metrics-csv snake/outputs/snake_lewm_continue_metrics.csv \
  --control-eval-seeds 6 \
  --control-eval-steps 180 \
  --control-eval-horizon 4 \
  --control-eval-mode planner
```

The continued run writes per-epoch metrics to:

```text
snake/outputs/snake_lewm_continue_metrics.csv
```

Plot the continued training loss and score with:

```bash
uv run python snake/plot_snake_training.py \
  --metrics snake/outputs/snake_lewm_continue_metrics.csv \
  --output snake/visualizations/snake_lewm_continue_training_curves.png
```

### Learned Control Algorithm

`SnakeLeWMController` is implemented in `run_snake_controller.py` and supports
two modes.

Policy mode:

```text
pixels_t -> encoder -> z_t -> policy_head(z_t) -> action logits
```

The legal action with the highest logit is selected, subject to a one-step
safety shield.

Planner mode:

1. Encode the current rendered frame to `z_t`.
2. Enumerate discrete action sequences of length `horizon`.
3. Roll each sequence through the learned latent predictor.
4. Decode predicted latents with probe heads.
5. Score each sequence by predicted progress toward the current food cell.
6. Add a small policy prior from `policy_head(z_t)`.
7. Execute the first action of the best sequence.
8. Re-render the true environment and replan next step.

Planner sequence score:

```text
score = max_t(100 * predicted_food_reached_t - toroidal_distance(predicted_head_t, food) - 0.03 * t)
      + 0.25 * policy_logit(first_action)
```

Safety shield:

- Before executing the selected action, the controller checks whether the action causes immediate self-collision in the real simulator.
- If it does, the controller falls back to another legal action.
- This prevents obviously fatal one-step actions but does not plan with the exact simulator.
- Videos using this controller should be described as `learned LeWM controller + immediate safety shield`.

Representative command:

```bash
uv run python snake/run_snake_controller.py \
  --controller lewm \
  --lewm-mode planner \
  --lewm-model snake/outputs/snake_lewm.pt \
  --horizon 5 \
  --steps 180 \
  --seed 16 \
  --video snake/visualizations/snake_lewm_planner_median_seed16.mp4 \
  --actions-out snake/outputs/snake_lewm_planner_median_seed16_actions.txt
```

### Evaluation Protocol

The learned controller is evaluated across fixed seeds, not just one video.

Metrics:

- Food score: number of food items eaten.
- Final length: initial length plus score.
- Death count: number of runs ending in self-collision or timeout.
- Steps survived.
- Mean, median, max, weak, median, and strong representative runs.

Initial 30-seed benchmark for `snake/outputs/snake_lewm.pt`:

```text
mean_score: 18.37
median_score: 20
max_score: 23
deaths: 10 / 30
weak: seed 0, score 3
median: seed 16, score 20
strong: seed 28, score 23
```

Continued checkpoint benchmark for `snake/outputs/snake_lewm_continued.pt`:

```text
mean_score: 18.83
median_score: 20
max_score: 27
deaths: 14 / 30
weak: seed 8, score 11
median: seed 23, score 20
strong: seed 28, score 27
```

Interpretation:

- Continued training reduced training loss and slightly improved mean/max score.
- Continued training also increased death rate.
- The continued checkpoint is not an unambiguous improvement, so both checkpoints should be compared depending on whether peak score or robustness matters more.

### Visualization Protocol

Dataset visualizations:

```bash
uv run python snake/visualize_snake_dataset.py \
  --data snake/data/snake.h5 \
  --output-dir visualizations \
  --count 4
```

This renders examples such as edge crossing, failure, high-score, and sample episodes.

LeWM prediction visualization:

```bash
uv run python snake/visualize_snake_lewm_predictions.py \
  --data snake/data/snake.h5 \
  --model snake/outputs/snake_lewm.pt \
  --output snake/visualizations/snake_lewm_prediction_teacher_forced.mp4 \
  --episode 206 \
  --horizon 120 \
  --teacher-forcing
```

Teacher-forced prediction re-encodes the true next frame at every step. This
tests one-step transition quality.

Open-loop prediction omits `--teacher-forcing` and feeds predicted latents back
into the model. Drift is expected and is why the controller replans every step.

Representative control videos:

- `snake/visualizations/snake_lewm_planner_weak_seed0.mp4`
- `snake/visualizations/snake_lewm_planner_median_seed16.mp4`
- `snake/visualizations/snake_lewm_planner_strong_seed28.mp4`
- `snake/visualizations/snake_lewm_showcase.mp4`
- `snake/visualizations/snake_lewm_continued_median_seed23.mp4`
- `snake/visualizations/snake_lewm_continued_strong_seed28.mp4`

The best single representative video for the initial checkpoint is:

```text
snake/visualizations/snake_lewm_planner_median_seed16.mp4
```

The best overview video is:

```text
snake/visualizations/snake_lewm_showcase.mp4
```

### Reproducibility Checklist

To reproduce the current experiment from scratch:

1. Generate `snake/data/snake.h5` with the latest dataset command above.
2. Render dataset videos with `visualize_snake_dataset.py`.
3. Train `snake/outputs/snake_lewm.pt` with the initial SnakeLeWM command.
4. Render prediction videos with `visualize_snake_lewm_predictions.py`.
5. Evaluate 30 seeds with `SnakeLeWMController` in planner mode.
6. Render weak, median, and strong videos.
7. Optionally continue training from `snake/outputs/snake_lewm.pt`.
8. Plot `snake/outputs/snake_lewm_continue_metrics.csv` with `snake/plot_snake_training.py`.
9. Compare original and continued checkpoints on the same fixed seeds.

### Key Caveats

- This is a LeWM-style hybrid controller, not pure canonical LeWM. The latent dynamics use LeWM's prediction plus SIGReg objective, but auxiliary probe and policy heads are used for practical control.
- The controller uses an immediate safety shield to avoid one-step self-collisions.
- Open-loop latent rollouts drift; the controller relies on frequent replanning.
- The exact simulator MPC remains a stronger non-learned baseline.
- Food score is noisy across seeds, so use multi-seed averages and representative videos rather than a single cherry-picked run.
