from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.sensor_stream import generate_sensor_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable predictive maintenance platform"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and compare ML models")
    train.add_argument("--data", required=True, type=Path, help="Path to CSV dataset")
    train.add_argument(
        "--target",
        default="failure_binary",
        choices=["failure_binary", "failure_type"],
        help="Prediction target",
    )
    train.add_argument("--reports-dir", default=Path("reports"), type=Path)
    train.add_argument("--tune", action="store_true", help="Run GridSearchCV tuning")
    train.add_argument("--test-size", default=0.2, type=float)
    train.add_argument("--random-state", default=42, type=int)

    simulate = subparsers.add_parser("simulate", help="Print simulated sensor rows")
    simulate.add_argument("--rows", default=5, type=int)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        from src.models.train import train_and_evaluate

        result = train_and_evaluate(
            data_path=args.data,
            target=args.target,
            reports_dir=args.reports_dir,
            tune=args.tune,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(result.metrics_table.to_string(index=False))
        print(f"\nBest model: {result.best_model_name}")
        print(f"Artifacts directory: {result.artifact_dir}")
    elif args.command == "simulate":
        print(generate_sensor_batch(args.rows).to_string(index=False))


if __name__ == "__main__":
    main()
