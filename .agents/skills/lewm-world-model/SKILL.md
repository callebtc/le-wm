---
name: lewm-world-model
description: Build, train, evaluate, visualize, and control systems with LeWorldModel-style latent world models. Use this skill whenever the user wants to create a world model, collect or convert datasets for action-conditioned dynamics, train LeWM/JEPA models from pixels or structured observations, run latent MPC/CEM/discrete planners, diagnose prediction drift, create rollout videos, compare learned control against scripted/expert baselines, or adapt the Snake workflow to a new simulator, robot, game, UI, sensor stream, or other environment.
---

# LeWM World Model Builder

Use this skill to build an end-to-end world-model pipeline around LeWorldModel (LeWM) or a LeWM-inspired latent dynamics model. The goal is not just to train a network, but to produce a usable control loop with datasets, diagnostics, prediction visualizations, and honest performance videos.

## Core Mental Model

LeWM is a reward-free, reconstruction-free JEPA world model:

```text
observation_t + action_t -> predicted latent_{t+1}
```

It learns an encoder and latent predictor:

```text
z_t = encoder(o_t)
z_hat_{t+1} = predictor(z_t, a_t)
loss = mse(z_hat_{t+1}, z_{t+1}) + lambda * SIGReg(z)
```

Important consequences:

- Rewards, scores, goals, and done flags are not required for core LeWM training.
- The dataset needs observations and aligned actions with broad dynamics coverage.
- Goals usually enter at planning/evaluation time as future observations or desired latent states.
- Open-loop rollouts drift; use short horizons and replan from real observations.
- Decoders and probes are diagnostics, not part of the core LeWM objective unless explicitly making a hybrid practical controller.

## When Starting A New Domain

First establish these pieces before training:

1. Environment/simulator interface: `reset(seed)`, `step(action)`, render/observe, done condition.
2. Action space: continuous, discrete, multi-discrete, or structured commands.
3. Observation type: pixels, low-dimensional state, sensor arrays, events, text/log state, or multimodal.
4. Control objective: goal image/state matching, score maximization, survival, reaching a target, imitation, or anomaly detection.
5. Dataset policy mix: random/exploratory, scripted, expert, failure cases, edge cases, and near-goal states.
6. Evaluation metrics: success rate, return/score, distance-to-goal, prediction error, survival time, collision/failure rate.

Ask one short question only if these are ambiguous enough to affect architecture or data collection.

## Dataset Requirements

For LeWM-compatible pixel training, export an HDF5 dataset with at least:

```text
pixels        uint8/float, shape (N, H, W, C) or compatible image frames
action        float, shape (N, A) or action-block vectors
ep_len        int, shape (num_episodes,)
ep_offset     int, shape (num_episodes,)
```

Strongly recommended extra columns:

```text
next_pixels       for visualization or custom trainers
state             privileged state/probes only, not core LeWM input
next_state        probe/diagnostic targets
reward            baseline/debugging only
done              baseline/debugging only
score             task metric
episode_idx       row episode id
step_idx          row step id
behavior_id       random/expert/scripted/mixed policy id
expert_action     optional policy-head/imitation label
event flags       collisions, wraps, contacts, resets, object pickups, etc.
```

For non-pixel inputs, keep the same episode structure and replace `pixels` with a domain-specific key such as `state`, `observation`, `sensor`, or `tokens`. Then replace the pixel encoder with an MLP, 1D CNN, transformer, graph encoder, or multimodal encoder that outputs a latent vector.

Dataset coverage matters more than expert optimality. Include:

- Successful trajectories.
- Failures and recoveries.
- Random exploration.
- Pseudo-expert/scripted behavior.
- Rare transitions and boundary conditions.
- Near-goal and near-failure states.
- Object interactions, collisions, wraps, contacts, mode switches, and resets where relevant.

Use fixed seeds and log dataset stats after generation: number of episodes, transitions, action distribution, failure count, event counts, score distribution, and frame shapes.

## Data Scale Guidance

For a smoke test, a few hundred episodes can validate the pipeline. For a useful controller, target thousands to tens of thousands of episodes or enough transitions to cover dynamics well.

The LeWM paper used 10k-20k episodes for several environments, 224x224 RGB frames, batch size 128, sub-trajectories of 4 frames, frame-skip 5, and around 10 epochs.

For small discrete games, many short diverse episodes are usually better than a few long expert episodes.

## Training Variants

### Canonical Pixel LeWM

Use when the project should follow the paper closely.

Inputs:

```text
pixels: (B, T, C, H, W)
actions: (B, T, A)
```

Model:

- Encoder: ViT-Tiny or smaller image encoder for simple domains.
- Projector: MLP with BatchNorm after the encoder latent.
- Predictor: action-conditioned transformer/MLP that predicts next latent.
- Regularizer: SIGReg on latent embeddings.

Loss:

```text
pred_loss = mse(predicted_next_latent, encoded_next_latent)
sigreg_loss = SIGReg(latents)
loss = pred_loss + lambda * sigreg_loss
```

Start with:

```text
latent_dim: 192
SIGReg projections: 256 for local/smoke, 1024 for paper-like runs
lambda: 0.05-0.1
predictor dropout: 0.1
history: 1-4 frames depending on whether velocity/direction is visible
```

If using this repo's generic training path, add a data config under `config/train/data/<task>.yaml` and run:

```bash
uv run python train.py data=<task> wandb.enabled=false trainer.max_epochs=10
```

### Practical Hybrid LeWM

Use when the user needs a working controller quickly.

Keep the core LeWM loss, but add thin heads on top of the learned latent for diagnostics or control:

- Probe heads for privileged variables: position, velocity, object state, food position, direction, contact, length.
- Policy head trained from expert/scripted labels if you have them.
- Risk/safety head if immediate failure labels are available.

Treat these heads as practical scaffolding. Be explicit that they are auxiliary and not pure LeWM.

### Structured Non-Pixel World Model

Use when pixels are unnecessary or too costly.

Replace the image encoder with a state encoder:

```text
z_t = encoder(state_t)
z_hat_{t+1} = predictor(z_t, action_t)
```

The same latent prediction + SIGReg idea applies. Add domain-specific probes for interpretability.

## Planning And Control

Choose the planner based on the action space.

Continuous actions:

- Use CEM/MPPI/gradient planners.
- Paper-style CEM: 300 candidates, 30 elites, horizon 5, 10-30 iterations, variance 1.
- Execute only a short action block, then replan.

Discrete actions:

- Do not directly use Gaussian CEM.
- Use categorical CEM, beam search, exhaustive short-horizon search, MCTS, or policy-guided search.
- For tiny action spaces, exhaustive horizon 4-8 can be practical.

Goal-conditioned LeWM planning:

```text
z_now = encoder(current_observation)
z_goal = encoder(goal_observation)
for candidate action sequence:
  z = z_now
  for action in sequence:
    z = predictor(z, action)
  cost = distance(z, z_goal)
execute first action or first action block
re-observe and replan
```

Score-maximization domains need a goal-generation strategy. Examples:

- Use future states from high-score trajectories as visual goals.
- Use one-step subgoals such as "object consumed", "target reached", or "safe next region".
- Use an outer symbolic/heuristic loop to propose goals, and the LeWM planner to reach them.
- Combine latent planner with a learned policy head and a one-step safety shield.

Safety shields are acceptable when the environment has obvious invalid actions. Explain them clearly so videos are not misrepresented as pure open-loop world-model control.

## Evaluation Protocol

Always evaluate more than one seed. Avoid judging from a cherry-picked video.

Minimum evaluation:

```text
N seeds or N held-out initial states
mean score/success
median score/success
best and worst run
failure/death/collision rate
mean episode length
```

Compare against baselines:

- Random policy.
- Simple scripted/greedy policy.
- Exact simulator MPC if available.
- Behavioral cloning/policy-only head.
- Learned world-model planner.

Representative videos:

- Weak run: shows failure mode.
- Median run: best single honest summary.
- Strong run: shows capability.
- Baseline comparison: random/greedy/exact MPC if useful.
- Showcase grid: 2x2 or side-by-side video with labels.

## Prediction Diagnostics

Create these visualizations before claiming the model works:

1. Dataset samples: videos from random, expert, failure, rare-event, and high-score episodes.
2. Teacher-forced one-step predictions: feed true current observation each step; compare predicted next state/decoded latent with actual next observation.
3. Open-loop rollout: feed model predictions back into the model; expect drift and measure when it starts.
4. Goal-reaching rollout: current, predicted trajectory, goal side by side.
5. Surprise/violation-of-expectation: teleport objects, change colors, alter physics, or inject impossible events and plot prediction error.
6. Latent probes: train linear/MLP probes to check whether important variables are recoverable.

For pixel LeWM, train a lightweight decoder only for visualization. Do not use reconstruction loss unless the user explicitly wants a generative model; the LeWM paper found reconstruction loss can hurt control.

## Metrics To Report

Training:

```text
pred_loss
sigreg_loss
total_loss
probe accuracies/MSE if probes exist
policy accuracy if policy head exists
device used: mps/cuda/cpu
training duration
dataset size
```

Prediction:

```text
one-step latent MSE
teacher-forced decoded/probe accuracy
open-loop horizon until drift
head/object/target localization accuracy if applicable
surprise spike for impossible events
```

Control:

```text
mean/median/best/worst score
success rate
failure/death rate
mean episode length
comparison to random/scripted/exact baselines
which video is representative and why
```

## Hardware Notes

Use `uv run` in this repo unless the user asks otherwise.

Apple Silicon:

- PyTorch exposes the Mac GPU as one logical `mps:0` device.
- There is no CUDA-style multi-GPU `DataParallel` across separate Apple GPU devices.
- Random compressed HDF5 reads can starve MPS; preload selected rows into RAM or cache arrays when feasible.
- Print backend availability at trainer startup.

CUDA:

- Use `cuda` if available.
- Use `DataParallel` or DDP only when multiple CUDA GPUs are visible and the training code supports it.

Always run a GPU smoke test:

```python
import torch
print(torch.backends.mps.is_available(), torch.cuda.is_available())
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1024, 1024, device=device)
print((x @ x).device)
```

## Common Failure Modes

- Dataset lacks rare events, so planner fails at boundaries or contacts.
- Model performs well teacher-forced but fails open-loop; solve with shorter horizon and replanning.
- Low-dimensional/simple domains conflict with high-dimensional SIGReg; tune `lambda` and latent dimension.
- Discrete actions are planned with continuous CEM incorrectly; use categorical/discrete search.
- Videos are cherry-picked; render weak/median/strong examples.
- Auxiliary heads are confused with pure LeWM; document what is core versus scaffolding.
- Pixel input is too ambiguous; add history frames, action blocks, or better rendering.
- The controller optimizes latent distance to impossible goals; sample reachable goals from held-out trajectories.

## Recommended End-To-End Workflow

Use this checklist when implementing a new domain:

1. Build or wrap the simulator/environment.
2. Add seedable reset and deterministic replay.
3. Implement random, scripted/expert, and failure-inducing controllers.
4. Export HDF5 with observations, actions, episode offsets, and diagnostics.
5. Print dataset stats and render sample videos.
6. Train a LeWM-style latent model from observations/actions.
7. Add probes or a decoder to inspect what the latent contains.
8. Render teacher-forced and open-loop prediction visualizations.
9. Implement an action-space-appropriate planner.
10. Evaluate over many seeds and compare against baselines.
11. Render weak, median, strong, and showcase videos.
12. Report caveats honestly: safety shields, auxiliary heads, open-loop drift, and baseline gap.

## Output Style When Reporting Results

When summarizing a world-model run, include:

- What was trained and whether it is canonical LeWM or hybrid LeWM.
- Dataset size, coverage highlights, and event counts.
- Device used and whether GPU acceleration was verified.
- Training metrics.
- Prediction metrics.
- Control metrics across multiple seeds.
- Exact artifact paths for models, datasets, action logs, and videos.
- Which video is the most representative and why.
- What still fails and the next concrete improvement.

Do not claim a model is optimal unless it is evaluated against strong baselines and consistently wins. Prefer "reasonably good", "median representative", "strong but not exact-MPC-level", or similar factual language.
