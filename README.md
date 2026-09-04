# Double Pendulum Balancer

A cart-pole double pendulum, simulated in PyBullet, trained to balance upright using DDPG (deep deterministic policy gradient) reinforcement learning.

The cart moves left and right along a rail. A two-link pendulum is attached to it. The goal is to keep both links balanced upright by applying horizontal force to the cart, and eventually to recover from a full fall by swinging the pendulum back up.

## Status

The agent currently learns fine balance control well: starting from a small angle near vertical, it keeps both links upright and the cart centered. It does not yet perform full swing-up recovery from a hanging start. That's the harder end of the curriculum described below and still in progress.

## How it works

### Environment

- PyBullet simulation of a cart on a rail with a two-link pendulum attached.
- Observation: cart position, cart velocity, sin/cos of each pole's angle, and each pole's angular velocity (8 values total). Angles are given as sin/cos rather than raw radians to avoid a discontinuity at +/-180 degrees, which matters once training includes large swings.
- Action: a single continuous force applied to the cart.
- Episodes end when the cart reaches the rail limits (a real, unrecoverable failure), or after a fixed number of steps. Falling over does not end the episode on its own; the reward function penalizes it, but the agent is allowed to keep trying to recover within the same episode.

### Agent

Standard DDPG: an actor network that outputs a continuous force, and a critic network that estimates the value of state-action pairs, each with a slowly-updated target network (Polyak averaging). Exploration is handled with an Ornstein-Uhlenbeck noise process added to the actor's output.

### Curriculum learning

Training starts with the pendulum reset close to vertical, with a tight failure angle. As the agent's performance improves (tracked as a rolling average of episode length), both the reset range and the failure angle are widened together, with a growing gap between them. This gives the agent room to fall and recover within an episode instead of just failing immediately. At full difficulty the pendulum resets from a full hang and the angle-based failure condition is effectively disabled, leaving swing-up and recovery as the only way to score well.

### Exploration scheduling

Exploration noise decays over training, but not on a fixed schedule. Decay only begins once the agent shows real, sustained performance at its current curriculum level, and gets partially reintroduced each time the curriculum advances to a harder level, since a harder task usually needs more exploration to find a working policy again.

## Project structure

```
src/
  config.py 	 all configurations of the project
  agent/         DDPG actor, critic, replay buffer
  physics/       PyBullet environment, state representation, reward
  trainer/       training loop, curriculum manager, exploration scheduler and noise
  checkpointer/  saving and loading model weights, curriculum and noise state
assets/
  urdf/          description for the cart and pendulum

main.py 		 main entry (train and run functions)
```

## Running it

```
git clone git@github.com:Alireza2317/pendulum-balancer.git
cd pendulum-balancer
uv sync
uv run main.py
```

`main.py` has two entry points: `train()` runs training and logs to TensorBoard, `run()` loads the latest checkpoint and renders the simulation so you can watch the current policy. Training can be stopped and resumed at any time; checkpoints save the network weights along with the current curriculum level and exploration state.

To watch training progress:

```
tensorboard --logdir logs
```

## Notes on training

This is a genuinely hard control problem. DDPG on a chaotic, underactuated system like this is slow to converge, and a lot of the tuning here has gone into just getting past a cold start where the agent never stumbles into a useful trajectory in the first place. Expect training to take a long time and to plateau for stretches before making visible progress.