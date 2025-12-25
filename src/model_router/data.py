"""Bounded arithmetic fixtures, exact answer checking, and replay ingestion."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
import random
import re

MAX_EXPRESSION_CHARS = 256
MAX_RECORDS = 20_000
MAX_JSONL_LINE_CHARS = 100_000
MAX_DATASET_CHARS = 32 * 1024 * 1024


def _tree(expression: str):
    if not isinstance(expression, str) or not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("Expression must be nonempty bounded text")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid arithmetic expression") from exc

    def check(node, depth=0):
        if depth > 16:
            raise ValueError("Expression nesting limit exceeded")
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10_000:
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            check(node.operand, depth + 1)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            check(node.left, depth + 1)
            check(node.right, depth + 1)
            return
        raise ValueError("Only bounded integer +, -, and * expressions are allowed")

    check(tree.body)
    return tree.body


def expression_structure(expression: str) -> str:
    """Canonical AST shape: ignore literal values, spelling and redundant parentheses."""

    def shape(node):
        if isinstance(node, ast.Constant):
            return ["integer"]
        if isinstance(node, ast.UnaryOp):
            return ["negate", shape(node.operand)]
        return [type(node.op).__name__, shape(node.left), shape(node.right)]

    return json.dumps(shape(_tree(expression)), separators=(",", ":"))


def answer(expression: str) -> str:
    def visit(node):
        if isinstance(node, ast.Constant):
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = a + b
            elif isinstance(node.op, ast.Sub):
                value = a - b
            else:
                value = a * b
            if abs(value) > 10**12:
                raise ValueError("Arithmetic magnitude limit exceeded")
            return value
        raise AssertionError("Unvalidated AST")

    return str(visit(_tree(expression)))


def features(expression: str) -> list[float]:
    _tree(expression)  # validates syntax without invoking the answer oracle
    numbers = re.findall(r"\d+", expression)
    depth = maximum = 0
    for character in expression:
        depth += character == "("
        depth -= character == ")"
        maximum = max(maximum, depth)
    return [
        1.0,
        len(expression) / 100,
        sum(expression.count(op) for op in "+-*") / 10,
        max(map(len, numbers), default=0) / 4,
        float("*" in expression),
        maximum / 4,
    ]


@dataclass(frozen=True)
class Response:
    model_id: str
    text: str
    cost_units: int
    latency_ms: float | None = None
    provenance: str = "SIMULATED"
    source_model: str | None = None
    recorded_at: str | None = None

    def __post_init__(self):
        if (
            type(self.model_id) is not str
            or self.model_id not in {"small", "large"}
            or not isinstance(self.text, str)
            or len(self.text) > 4096
        ):
            raise ValueError("Invalid response identity or text")
        if type(self.cost_units) is not int or not 1 <= self.cost_units <= 100_000:
            raise ValueError("Cost must be positive integer units")
        if self.latency_ms is not None and (
            type(self.latency_ms) not in (float, int) or not 0 <= self.latency_ms <= 10**9
        ):
            raise ValueError("Latency must be measured nonnegative milliseconds or null")
        if type(self.provenance) is not str or self.provenance not in {"SIMULATED", "RECORDED"}:
            raise ValueError("Response provenance must be SIMULATED or RECORDED")
        if self.provenance == "SIMULATED" and any(
            value is not None for value in (self.latency_ms, self.source_model, self.recorded_at)
        ):
            raise ValueError("Simulated fixtures must not invent measured latency or recorded model metadata")
        if self.provenance == "RECORDED":
            if not isinstance(self.source_model, str) or not self.source_model or len(self.source_model) > 512:
                raise ValueError("Recorded responses require an exact source model/checkpoint identity")
            try:
                timestamp = datetime.fromisoformat(self.recorded_at)
            except (TypeError, ValueError) as exc:
                raise ValueError("Recorded responses require an ISO timestamp") from exc
            if timestamp.tzinfo is None:
                raise ValueError("Recorded timestamps must include a timezone")


@dataclass(frozen=True)
class Record:
    case_id: str
    template_id: str
    split: str
    expression: str
    responses: tuple[Response, ...]

    def __post_init__(self):
        if (
            any(type(value) is not str or not 1 <= len(value) <= 128 for value in (self.case_id, self.template_id))
            or type(self.split) is not str
            or self.split not in {"train", "calibration", "test"}
        ):
            raise ValueError("Record requires ID, template group, and recognized split")
        answer(self.expression)
        if (
            not isinstance(self.responses, tuple)
            or len(self.responses) != 2
            or not all(isinstance(row, Response) for row in self.responses)
            or {row.model_id for row in self.responses} != {"small", "large"}
        ):
            raise ValueError("Each record requires exactly one small and one large response")

    def response(self, model_id: str) -> Response:
        return next(row for row in self.responses if row.model_id == model_id)


def correct(record: Record, model_id: str) -> bool:
    text = record.response(model_id).text.strip()
    if re.fullmatch(r"-?\d{1,14}(?:/[1-9]\d{0,13})?", text) is None:
        return False
    try:
        return Fraction(text) == Fraction(answer(record.expression))
    except (ValueError, ZeroDivisionError):
        return False


def generate(seed: int = 17) -> list[Record]:
    rng = random.Random(seed)
    records = []
    patterns: set[str] = set()

    def pattern(leaves):
        if leaves == 1:
            return "{}"
        left = rng.randint(1, leaves - 1)
        return f"({pattern(left)} {rng.choice(('+', '-', '*'))} {pattern(leaves - left)})"

    for template in range(60):
        # Unique expression structures, not merely example IDs, are held out.
        while True:
            leaves = rng.randint(2, 6)
            skeleton = pattern(leaves)
            if skeleton not in patterns:
                patterns.add(skeleton)
                break
        bucket = template % 10
        split = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
        for variant in range(3):
            upper = rng.choice((9, 99, 999)) if leaves <= 3 else 9
            expression = skeleton.format(*(rng.randint(1, upper) for _ in range(leaves)))
            expected = int(answer(expression))
            vector = features(expression)
            success_logit = 4.5 - vector[2] * 10 * 0.7 - vector[3] * 4 * 0.45 - vector[4] * 1.1 - vector[5] * 4 * 0.20
            small_probability = 1 / (1 + math.exp(-success_logit))
            small = expected if rng.random() < small_probability else expected + rng.choice((-3, -1, 1, 2))
            large = expected if rng.random() < 0.95 else expected + 1
            group = hashlib.sha256(skeleton.encode()).hexdigest()[:16]
            records.append(
                Record(
                    f"t{template:02d}-v{variant}",
                    f"structure-{group}",
                    split,
                    expression,
                    (
                        Response("small", str(small), 1 + len(expression) // 20),
                        Response("large", str(large), 7 + len(expression) // 15),
                    ),
                )
            )
    validate_splits(records)
    return records


def validate_splits(records: list[Record]) -> None:
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= MAX_RECORDS
        or not all(isinstance(row, Record) for row in records)
    ):
        raise ValueError("Dataset must be nonempty and bounded")
    identifiers, groups, structures = set(), {}, {}
    for row in records:
        if row.case_id in identifiers:
            raise ValueError("Duplicate case ID")
        identifiers.add(row.case_id)
        if row.template_id in groups and groups[row.template_id] != row.split:
            raise ValueError("Template leakage across data splits")
        groups[row.template_id] = row.split
        structure = expression_structure(row.expression)
        if structure in structures and structures[structure] != row.split:
            raise ValueError("Expression-structure leakage across data splits")
        structures[structure] = row.split
    if {row.split for row in records} != {"train", "calibration", "test"}:
        raise ValueError("Dataset must include train, calibration, and test splits")


def serialized(records: list[Record]) -> str:
    return "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in records)


def dataset_hash(records: list[Record]) -> str:
    return hashlib.sha256(serialized(records).encode()).hexdigest()


def strict_json(text: str):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number is unsupported: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object field")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def load(path: str) -> list[Record]:
    result = []
    total = 0
    with open(path, encoding="utf-8") as source:
        while True:
            line = source.readline(MAX_JSONL_LINE_CHARS + 1)
            if not line:
                break
            total += len(line)
            if len(line) > MAX_JSONL_LINE_CHARS or total > MAX_DATASET_CHARS:
                raise ValueError("Dataset character limit exceeded")
            if not line.strip():
                continue
            if len(result) >= MAX_RECORDS:
                raise ValueError("Dataset record limit exceeded")
            row = strict_json(line)
            if not isinstance(row, dict) or not isinstance(row.get("responses"), list) or len(row["responses"]) != 2:
                raise ValueError("Dataset records must contain exactly two response objects")
            try:
                row["responses"] = tuple(Response(**item) for item in row["responses"])
                result.append(Record(**row))
            except (TypeError, KeyError) as exc:
                raise ValueError("Invalid record or response schema") from exc
    validate_splits(result)
    return result
