# Pendulum Balancer

A TD3 reinforcement-learning agent for controlling a double pendulum on a cart,
using TensorFlow and PyBullet. Training uses curriculum learning and periodic
headless evaluation; playback opens a rendered simulation.

## Setup

This project uses **uv** to manage Python and dependencies. From the repository root:

```bash
uv sync
```

## Train

```bash
# Use Config.max_episodes
uv run python main.py train --run-name experiment

# Override the episode count
uv run python main.py train --run-name experiment --episodes 100

# Resume for 100 additional episodes
uv run python main.py train --run-name experiment --episodes 100 --resume runs/experiment/checkpoints/ckpt-100
```

Add `--reset-replay-buffer` when resuming to clear the restored replay data.
Checkpoint paths are prefixes: omit `.index` and `.data-*` suffixes. Keep the
matching replay-buffer and curriculum `.pkl` files alongside checkpoints for resume.

Each run saves checkpoints, TensorBoard logs, evaluation CSVs, `config.json`, and
`manifest.json` under `runs/<run-name>/`. By default, evaluation and checkpointing
happen every five episodes; a checkpoint is also saved at the final episode.

```bash
uv run tensorboard --logdir runs/experiment/logs
```

## Run the simulation

```bash
uv run python main.py run --checkpoint runs/experiment/checkpoints/ckpt-100
```

Playback requires a graphical display. With the simulation window focused,
press `R` or use the on-screen reset button to reset the environment. Press
`Ctrl+C` to stop. Omitting
`--checkpoint` loads the latest checkpoint from the root `checkpoints/` directory,
not from `runs/`.

### Demo
<p align="center">
  <img width="720" height="480" alt="pend-balancer" src="https://github.com/user-attachments/assets/ef48bc68-c924-4744-8f3b-84fcc2ec2886" />
</p>



## Configuration

Configure training, physics, rewards, curriculum, exploration, evaluation frequency,
and checkpoint frequency through `Config` in [src/config.py](src/config.py).
The CLI provides overrides for episode count and options for run name, checkpoint
selection, and replay-buffer reset. Saved `config.json` files record run settings;
the CLI uses the current `Config` defaults, including when resuming.

Network architecture is defined in `src/agent/models.py`; playback's reset range
is currently set in `src/playback.py`.

## Standalone evaluation and help

```bash
uv run python -m src.evaluation.grid --checkpoint runs/experiment/checkpoints/ckpt-100 --level 0.0 --mode full --grid-size 5
uv run python main.py --help
uv run python main.py train --help
uv run python main.py run --help
```

Standalone evaluation writes CSVs to `artifacts/evaluations/`. Use `--mode sentinel`
for five test cases, or `--mode full` for a square grid with an odd size of at least 3.
