"""Offline training/evaluation CLI. No implicit downloads or API requests."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .core import evaluate, evaluate_sweep, load_router, train
from .data import generate, load, serialized


def emit(value, output):
    text = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(
        description="Learn and evaluate budget-aware routing over labeled response records."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate", "generate", "train", "sweep"):
        command = commands.add_parser(name)
        command.add_argument("--output")
        command.add_argument("--records", help="JSONL response records; default explicitly simulated fixtures")
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--budget-units", type=int, default=180)
        if name in {"evaluate", "demo", "sweep"}:
            command.add_argument("--router", help="Previously trained JSON router artifact")
        if name == "sweep":
            command.add_argument("--budgets", type=int, nargs="+", default=[0, 40, 80, 180])
            command.add_argument("--orderings", type=int, default=5)
    for name in ("bandit-demo", "bandit-generate", "bandit-evaluate"):
        command = commands.add_parser(name, help="Stateless per-request logged-bandit experiment")
        command.add_argument("--output")
        command.add_argument("--seed", type=int, default=17)
        command.add_argument("--folds", type=int, default=5)
        if name == "bandit-evaluate":
            command.add_argument("--logs", required=True)
            command.add_argument(
                "--policy", choices=("small", "large", "adaptive", "soft_adaptive", "admit"), default="adaptive"
            )
        else:
            command.add_argument("--samples", type=int, default=1500)
            command.add_argument("--epsilon", type=float, default=0.3)
    args = parser.parse_args()
    if args.command.startswith("bandit-"):
        from .bandit import bandit_demo, evaluate_logged, load_logged, synthetic_log, target_policy

        try:
            if args.command == "bandit-demo":
                emit(bandit_demo(args.samples, args.seed, args.epsilon, args.folds), args.output)
            elif args.command == "bandit-generate":
                events, _ = synthetic_log(args.samples, args.seed, args.epsilon)
                text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in events)
                if args.output:
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8")
                else:
                    print(text, end="")
            else:
                report = evaluate_logged(
                    load_logged(args.logs), target_policy(args.policy), folds=args.folds, seed=args.seed
                )
                emit(report, args.output)
                if not report["usable"]:
                    raise SystemExit(1)
        except ValueError as exc:
            parser.error(str(exc))
        return
    records = load(args.records) if args.records else generate(args.seed)
    if args.command == "generate":
        text = serialized(records)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    elif args.command == "train":
        emit(train(records, args.budget_units).to_dict(), args.output)
    else:
        router = load_router(args.router) if args.router else None
        report = (
            evaluate_sweep(
                records,
                args.budgets,
                reference_budget=args.budget_units,
                orderings=args.orderings,
                seed=args.seed,
                router=router,
            )
            if args.command == "sweep"
            else evaluate(records, args.budget_units, router)
        )
        if args.command == "demo":
            report["policies"] = [
                {key: value for key, value in policy.items() if key != "decisions"} for policy in report["policies"]
            ]
        emit(report, args.output)


if __name__ == "__main__":
    main()
