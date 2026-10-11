"""
metrics/instruction_mix.py — Instruction Mix Analysis

Categorizes every PTX instruction into functional groups and computes
the percentage breakdown. This reveals what a kernel is "spending time on."

CATEGORIES:
    ARITHMETIC   — add, sub, mul, div, fma, mad, abs, neg, min, max, sqrt, rcp
    MEMORY       — ld, st, prefetch, isspacep, cvta
    MOVEMENT     — mov, cvt, selp, shfl (warp shuffle)
    COMPARISON   — setp (set predicate)
    BRANCH       — bra, ret, exit, call, brx
    SPECIAL      — bar (barrier), atom (atomic), red (reduction), vote, match
    OTHER        — everything else

WHY INSTRUCTION MIX MATTERS:
    High ARITHMETIC %  → compute-bound kernel → GPU is working hard (good)
    High MEMORY %      → memory-bound kernel  → may need better data layout
    High BRANCH %      → control-bound kernel → potential divergence issues
    High MOVEMENT %    → register shuffling   → compiler may not be optimal

INTERPRETING THE MIX:
    For GPU kernels, a healthy mix is typically:
        ARITHMETIC  > 30%   — GPU doing real compute work
        MEMORY      < 25%   — not spending too much time on loads/stores
        BRANCH      < 10%   — minimal control flow (GPU hates branching)
"""

from dataclasses import dataclass, field
from typing import Dict, List
from ..ir import Instruction, PTXNode


# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = {
    'ARITHMETIC': {
        'add', 'sub', 'mul', 'div', 'fma', 'mad', 'abs', 'neg',
        'min', 'max', 'sqrt', 'rsqrt', 'rcp', 'sin', 'cos', 'lg2',
        'ex2', 'rem', 'sad', 'popc', 'clz', 'bfind', 'brev',
    },
    'MEMORY': {
        'ld', 'st', 'prefetch', 'prefetchu', 'isspacep', 'cvta',
        'atom', 'red',
    },
    'MOVEMENT': {
        'mov', 'cvt', 'selp', 'shfl', 'prmt',
    },
    'COMPARISON': {
        'setp', 'set',
    },
    'BRANCH': {
        'bra', 'ret', 'exit', 'call', 'brx', 'trap',
    },
    'SPECIAL': {
        'bar', 'membar', 'vote', 'match', 'activemask', 'nanosleep',
        'cp', 'mbarrier', 'wgmma', 'fence',
    },
}

# Build reverse lookup: base_op → category
_OP_TO_CATEGORY: Dict[str, str] = {}
for _cat, _ops in CATEGORIES.items():
    for _op in _ops:
        _OP_TO_CATEGORY[_op] = _cat


def _categorize(opcode: str) -> str:
    """Map an opcode to its category."""
    base = opcode.split('.')[0].lower()
    return _OP_TO_CATEGORY.get(base, 'OTHER')


# ── Metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class InstructionMixMetrics:
    """Instruction mix breakdown for one PTX kernel."""

    counts: Dict[str, int] = field(default_factory=lambda: {
        cat: 0 for cat in list(CATEGORIES.keys()) + ['OTHER']
    })
    total: int = 0

    # Detailed sub-counts for memory category
    global_mem_ops: int = 0
    shared_mem_ops: int = 0

    def percentage(self, category: str) -> float:
        """Return percentage of instructions in this category."""
        if self.total == 0:
            return 0.0
        return self.counts.get(category, 0) / self.total * 100

    def bar_chart(self, category: str, width: int = 20) -> str:
        """ASCII progress bar for a category."""
        pct = self.percentage(category)
        filled = int(pct / 100 * width)
        return '█' * filled + '░' * (width - filled)

    @property
    def dominant_category(self) -> str:
        """The category with the most instructions."""
        if not self.counts:
            return "UNKNOWN"
        return max(self.counts, key=lambda k: self.counts[k])

    @property
    def compute_intensity(self) -> str:
        """
        Ratio of arithmetic to memory ops — higher = more compute-bound.
        Compute-bound kernels benefit from algorithmic optimization.
        Memory-bound kernels benefit from data layout optimization.
        """
        arith = self.counts.get('ARITHMETIC', 0)
        mem   = self.counts.get('MEMORY', 0)
        if mem == 0:
            return "COMPUTE ONLY (no memory ops)"
        ratio = arith / mem
        if ratio > 4:   return f"COMPUTE-BOUND  (ratio {ratio:.1f}:1)"
        if ratio > 1:   return f"BALANCED       (ratio {ratio:.1f}:1)"
        return         f"MEMORY-BOUND   (ratio {ratio:.1f}:1)"


# ── Analysis function ─────────────────────────────────────────────────────────

def analyze_instruction_mix(nodes: List[PTXNode]) -> InstructionMixMetrics:
    """
    Categorize all instructions in the node list.

    Args:
        nodes: parsed PTXNode list

    Returns:
        InstructionMixMetrics with full breakdown
    """
    metrics = InstructionMixMetrics()

    for node in nodes:
        if not isinstance(node, Instruction):
            continue

        category = _categorize(node.opcode)
        metrics.counts[category] = metrics.counts.get(category, 0) + 1
        metrics.total += 1

        # Track global vs shared memory ops
        op = node.opcode.lower()
        if 'global' in op:  metrics.global_mem_ops += 1
        if 'shared' in op:  metrics.shared_mem_ops += 1

    return metrics
