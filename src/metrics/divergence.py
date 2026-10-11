"""
metrics/divergence.py — Branch Divergence Analysis

Detects patterns in PTX that can cause warp divergence — one of the
most important GPU performance problems.

KEY CONCEPT — What is a Warp?
    GPU threads run in groups of 32 called WARPS.
    All 32 threads in a warp execute the SAME instruction simultaneously.
    This is called SIMT (Single Instruction, Multiple Threads).

WHAT IS DIVERGENCE?
    When threads in the same warp take DIFFERENT branches:

        if (threadIdx.x % 2 == 0) {
            // even threads come here
        } else {
            // odd threads come here
        }

    The GPU must serialize these:
        Step 1: Run "even" branch, mask out odd threads  → 50% efficiency
        Step 2: Run "odd"  branch, mask out even threads → 50% efficiency
    Result: 2x slower than no branching!

HOW WE DETECT IT IN PTX:
    Every conditional branch in PTX uses two instructions:

        setp.ge.u32  %p0, %r1, %r0    ← COMPARISON: sets predicate register
        @%p0 bra     END;              ← CONDITIONAL BRANCH: uses predicate

    Your tool finds every setp+bra pair and each one is a potential
    divergence point.

DIVERGENCE RISK LEVELS:
    0 conditional branches → NONE
    1-2                    → LOW
    3-5                    → MEDIUM
    6+                     → HIGH
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple
from ..ir import Instruction, RawLine, PTXNode


@dataclass
class DivergencePoint:
    """A single detected potential divergence location."""
    lineno:     int          # line number of the conditional branch
    branch_raw: str          # raw PTX text of the branch instruction
    pred_reg:   str          # predicate register used (%p0, %p1, etc.)
    setp_raw:   str = ""     # raw PTX of the setp that set this predicate


@dataclass
class DivergenceMetrics:
    """Branch divergence analysis results."""

    conditional_branches: int = 0       # total @%p bra instructions
    unconditional_branches: int = 0     # plain bra instructions
    predicated_instructions: int = 0    # any @%p instruction (not just bra)
    setp_count: int = 0                 # number of setp instructions
    divergence_points: List[DivergencePoint] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        n = self.conditional_branches
        if n == 0:   return "NONE"
        if n <= 2:   return "LOW"
        if n <= 5:   return "MEDIUM"
        return "HIGH"

    @property
    def risk_color_hint(self) -> str:
        """Text symbol for risk level (for terminal display)."""
        level = self.risk_level
        if level == "NONE":   return "✓"
        if level == "LOW":    return "~"
        if level == "MEDIUM": return "!"
        return "!!"

    @property
    def recommendation(self) -> str:
        level = self.risk_level
        if level == "NONE":
            return "No conditional branches — zero divergence risk."
        if level == "LOW":
            return "Few conditional branches — likely low impact on performance."
        if level == "MEDIUM":
            return ("Consider restructuring branches so threads in the same "
                    "warp take the same path (branch on blockIdx, not threadIdx).")
        return ("High divergence risk. Profile with Nsight Compute. "
                "Consider warp-level primitives (__ballot_sync, __any_sync) "
                "or restructure algorithm to eliminate per-thread branching.")


# ── Regex helpers ─────────────────────────────────────────────────────────────

_RE_PRED_BRANCH  = re.compile(r'@(!?)(%p\w*)\s+bra', re.IGNORECASE)
_RE_PRED_ANY     = re.compile(r'@(!?)(%p\w*)\s+\w')
_RE_SETP         = re.compile(r'^\s*setp\b', re.IGNORECASE)
_RE_BRA          = re.compile(r'^\s*(?:@\S+\s+)?bra\b', re.IGNORECASE)
_RE_PRED_REG     = re.compile(r'%p\w+')


def analyze_divergence(nodes: List[PTXNode]) -> DivergenceMetrics:
    """
    Scan PTX nodes for branch divergence indicators.

    Strategy:
    1. Count all setp instructions (comparisons that feed branches)
    2. Find all @%p bra patterns (conditional branches)
    3. For each conditional branch, find the setp that set that predicate
    4. Build DivergencePoint records for reporting

    Args:
        nodes: parsed PTXNode list

    Returns:
        DivergenceMetrics with full divergence analysis
    """
    metrics = DivergenceMetrics()

    # Build map: predicate_register → setp instruction text
    # (most recent setp that wrote to this pred reg)
    pred_to_setp: dict = {}

    for node in nodes:
        if not isinstance(node, Instruction):
            continue

        raw = node.raw

        # Count setp instructions and track which pred reg they set
        if _RE_SETP.match(raw):
            metrics.setp_count += 1
            if node.dest:
                pred_to_setp[node.dest] = raw.strip()

        # Count any @%p instruction (predicated execution)
        if _RE_PRED_ANY.search(raw):
            metrics.predicated_instructions += 1

        # Detect conditional branches: @%p0 bra TARGET
        m = _RE_PRED_BRANCH.search(raw)
        if m:
            pred_reg = m.group(2)
            metrics.conditional_branches += 1
            metrics.divergence_points.append(DivergencePoint(
                lineno=node.lineno,
                branch_raw=raw.strip(),
                pred_reg=pred_reg,
                setp_raw=pred_to_setp.get(pred_reg, "(setp not found in scope)"),
            ))
            continue

        # Plain unconditional branch
        if _RE_BRA.match(raw) and not _RE_PRED_ANY.search(raw):
            metrics.unconditional_branches += 1

    return metrics
