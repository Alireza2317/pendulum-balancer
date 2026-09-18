import argparse
from pathlib import Path


def positive_int(value: str) -> int:
	try:
		result = int(value)
	except ValueError:
		raise argparse.ArgumentTypeError("must be a positive integer") from None
	if result <= 0:
		raise argparse.ArgumentTypeError("must be a positive integer")
	return result


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Train or run the pendulum agent.")
	commands = parser.add_subparsers(dest="command", required=True)

	train_parser = commands.add_parser("train", help="Train the pendulum agent.")
	train_parser.add_argument("--episodes", type=positive_int, required=True)
	train_parser.add_argument("--run-name", required=True)
	train_parser.add_argument("--resume", type=Path, default=None)
	train_parser.add_argument(
		"--reset-replay-buffer",
		action="store_true",
		help="Clear restored replay data after loading a checkpoint.",
	)

	run_parser = commands.add_parser("run", help="Run the rendered simulation.")
	run_parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help="Checkpoint prefix; defaults to the latest checkpoint in checkpoints/.",
	)

	args = parser.parse_args(argv)
	if args.command == "train":
		from src.training import train

		train(
			episodes=args.episodes,
			run_name=args.run_name,
			resume=args.resume,
			reset_replay_buffer=args.reset_replay_buffer,
		)
	else:
		from src.playback import run

		run(checkpoint=args.checkpoint)
