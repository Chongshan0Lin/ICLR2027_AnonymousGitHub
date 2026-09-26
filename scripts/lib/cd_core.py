#!/usr/bin/env python3
"""Countdown solver, entrance families, prompt and parsing.

The solver, the entrance definition and the completion parser are ported from
the reference implementation (ershiyidian/early-branch-locking), not reinvented,
because the plan requires the entrance family to carry the same meaning there
and here.

An **entrance family** is `(first integer operand, first arithmetic operator)`
found in the generated reasoning, scanning stopped at </think>, <feasible> or
<answer> so that an expression inside the answer cannot be mistaken for a
reasoning entrance. That is the reference's definition; note it is a single
operand plus an operator, not an operand pair.
"""
from __future__ import annotations

import ast
import operator
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------- prompt
SYSTEM = ("You are a helpful assistant. You first thinks about the reasoning "
          "process in the mind and then provides the user with the answer.")
USER = ("Using the numbers {nums}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) one or multiple "
        "times but each number can only be used once. Show your work in "
        "<think> </think> tags. And return the final equation in <answer> "
        "</answer> tags, for example <answer> (1 + 2) / 3 </answer>. Think "
        "step by step inside <think> tags.")
ASSISTANT_PREFIX = "Let me solve this step by step.\n<think>"


def build_prompt(tok, nums, target) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(nums=list(nums),
                                                    target=target)},
            {"role": "assistant", "content": ASSISTANT_PREFIX}]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   continue_final_message=True)


# ----------------------------------------------------------------- solver
@lru_cache(maxsize=None)
def can_reach(nums: Tuple[int, ...], target: int) -> bool:
    """Reference brute-force feasibility: integer ops only, exact division."""
    n = len(nums)
    if n == 0:
        return False
    if n == 1:
        return nums[0] == target
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(n) if k not in (i, j)]
            res = [a + b, a * b, a - b, b - a]
            if b != 0 and a % b == 0:
                res.append(a // b)
            if a != 0 and b % a == 0:
                res.append(b // a)
            for r in res:
                if can_reach(tuple(sorted(rest + [r])), target):
                    return True
    return False


@lru_cache(maxsize=None)
def count_solutions(nums: Tuple[int, ...], target: int) -> int:
    """Number of distinct reduction sequences reaching the target.

    Counted over ordered (pair, operator) choices with the multiset of
    remaining values carried forward, so it measures how many ways the search
    can succeed below a node -- which is what "downstream solution
    multiplicity" means for this experiment. It is not a count of distinct
    algebraic expressions; two orderings that collapse to the same formula are
    counted separately, consistently for every branch.
    """
    n = len(nums)
    if n == 1:
        return 1 if nums[0] == target else 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(n) if k not in (i, j)]
            res = [a + b, a * b, a - b, b - a]
            if b != 0 and a % b == 0:
                res.append(a // b)
            if a != 0 and b % a == 0:
                res.append(b // a)
            for r in res:
                total += count_solutions(tuple(sorted(rest + [r])), target)
    return total


def enumerate_families(nums: List[int], target: int) -> List[Dict]:
    """Solver-feasible entrance families for one problem.

    A family is keyed `f"{operand}{op}"` to match the reference detector, which
    sees only the first operand and the first operator. Several concrete moves
    can therefore share one family -- 8*3 and 8*7 are both "8*" -- so the
    family's multiplicity is the sum over the moves it covers.
    """
    fam: Dict[str, Dict] = {}
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            rest = [nums[k] for k in range(n) if k not in (i, j)]
            for op in "+-*/":
                if op == "+":
                    r = a + b
                elif op == "-":
                    r = a - b
                elif op == "*":
                    r = a * b
                else:
                    if b == 0 or a % b != 0:
                        continue
                    r = a // b
                m = count_solutions(tuple(sorted(rest + [r])), target)
                if m <= 0:
                    continue
                key = f"{a}{op}"
                e = fam.setdefault(key, {"family_id": key, "operator": op,
                                         "first_operand": a, "moves": [],
                                         "solution_multiplicity": 0})
                e["moves"].append({"a": a, "b": b, "result": r,
                                   "multiplicity": m})
                e["solution_multiplicity"] += m
    tot = sum(v["solution_multiplicity"] for v in fam.values())
    for v in fam.values():
        v["fraction_of_valid_solutions"] = (v["solution_multiplicity"] / tot
                                            if tot else 0.0)
    return sorted(fam.values(), key=lambda v: -v["solution_multiplicity"])


# ----------------------------------------------------------- entrance detect
FIRST_ARITH = re.compile(r"(?<![\w.])(?P<first>-?\d+(?:\.\d+)?)\s*"
                         r"(?P<op>[+\-*/×÷])")
STOP_MARKERS = ("</think>", "<feasible>", "<answer>")


def find_entrance(generated: str):
    """(family, first_operand, operator) or (None, None, None)."""
    text = str(generated or "")
    stop = len(text)
    for m in STOP_MARKERS:
        k = text.lower().find(m.lower())
        if k >= 0:
            stop = min(stop, k)
    text = text[:stop]
    mt = FIRST_ARITH.search(text)
    if mt is None:
        return None, None, None
    raw = mt.group("first")
    if "." in raw:
        return None, None, None
    try:
        first = int(raw)
    except ValueError:
        return None, None, None
    op = {"×": "*", "÷": "/"}.get(mt.group("op"), mt.group("op"))
    return f"{first}{op}", first, op


# ----------------------------------------------------------------- answers
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
ANSWER_OPEN = re.compile(r"<answer>\s*(.*)", re.S | re.I)
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def extract_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text or "")
    if m:
        return m.group(1).strip()
    m = ANSWER_OPEN.search(text or "")
    if m:
        v = m.group(1).split("</answer>")[0].split("\n")[0].strip()
        return v or None
    return None


def evaluate_expression(expr: str, nums: List[int], target: int) -> Dict:
    """(valid_expression, reaches_target, value) under the reference rules."""
    out = {"parsed": False, "uses_numbers": False, "value": None,
           "reaches_target": False}
    if not expr:
        return out
    e = expr.replace("=", " ").strip()
    e = re.sub(r"[^0-9+\-*/(). ]", "", e).strip()
    if not e:
        return out
    consts: List[int] = []

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BinOp):
            t = type(node.op)
            if t not in _BIN:
                raise ValueError("op")
            return _BIN[t](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd,
                                                                  ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.Constant) and isinstance(node.value,
                                                         (int, float)):
            if isinstance(node.value, int):
                consts.append(int(abs(node.value)))
            return float(node.value)
        raise ValueError("node")

    try:
        val = ev(ast.parse(e, mode="eval"))
    except Exception:
        return out
    out["parsed"] = True
    out["value"] = val
    out["uses_numbers"] = sorted(consts) == sorted(int(x) for x in nums)
    out["reaches_target"] = (out["uses_numbers"]
                             and abs(val - float(target)) < 1e-6)
    return out
