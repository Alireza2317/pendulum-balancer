from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
	total_reward: float
	steps: int
	balance_fraction: float
	failure_reason: str | None


@dataclass(frozen=True)
class AgentMetrics:
	actor_loss: float
	critic_loss: float
	q_vals_avg: float
