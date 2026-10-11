"""
metrics/registers.py — Register Usage Analysis

Analyzes how many registers a PTX kernel declares, categorized by type.
Also computes register pressure — high register count limits GPU occupancy.

KEY CONCEPT:
    Each GPU Streaming Multiprocessor (SM) has a fixed register file
    (e.g. 65,536 registers on Ampere GPUs). These are shared across all
    active threads. More registers per thread = fewer threads can run
    simultaneously = lower occupancy = lower performance.

    Rule of thumb:
        < 32 registers  → LOW pressure  (good)
        32-63 registers → MEDIUM pressure
        64+ registers   → HIGH pressure (may cause register spilling)

Register types in PTX:
    %f   / .f32  — 32-bit float
    %fd  / .f64  — 64-bit double
    %r   / .u32  — 32-bit integer
    %rd  / .u64  — 64-bit address/integer
    %p   / .pred — predicate (condition) register
    %b   / .b32  — 32-bit bitfield
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class RegisterMetrics:
    """Complete register usage metrics for one PTX kernel."""

    # Per-type register counts: {"f32": 8, "u64": 4, "pred": 2, ...}
    by_type: Dict[str, int] = field(default_factory=dict)

    # Human-readable prefix map: {"f32": "%f", "u64": "%rd", ...}
    prefixes: Dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Total number of registers declared across all types."""
        return sum(self.by_type.values())

    @property
    def float_regs(self) -> int:
        return self.by_type.get('f32', 0) + self.by_type.get('f16', 0)

    @property
    def double_regs(self) -> int:
        return self.by_type.get('f64', 0)

    @property
    def int_regs(self) -> int:
        return sum(v for k, v in self.by_type.items()
                   if k.startswith('u') or k.startswith('s') or k.startswith('b'))

    @property
    def addr_regs(self) -> int:
        return self.by_type.get('u64', 0) + self.by_type.get('b64', 0)

    @property
    def pred_regs(self) -> int:
        return self.by_type.get('pred', 0)

    @property
    def pressure_level(self) -> str:
        """
        Classify register pressure.
        Lower is better — more threads can run simultaneously.
        """
        t = self.total
        if t == 0:   return "UNKNOWN"
        if t < 24:   return "LOW (excellent occupancy)"
        if t < 32:   return "LOW"
        if t < 48:   return "MEDIUM"
        if t < 64:   return "HIGH"
        return "VERY HIGH (risk of register spilling)"

    @property
    def max_threads_per_sm(self) -> int:
        """
        Estimate max concurrent threads on an SM with 65536 registers.
        Real GPUs: Ampere = 65536, Volta = 65536, Pascal = 65536.
        """
        if self.total == 0:
            return 0
        # Round up to nearest power of 2 (GPU allocates in chunks)
        alloc = self.total
        for p in [2, 4, 8, 16, 32, 64, 128]:
            if alloc <= p:
                alloc = p
                break
        return min(65536 // alloc, 2048)  # 2048 = max threads per SM


def analyze_registers(reg_directives: List[Tuple[str, str, int]]) -> RegisterMetrics:
    """
    Build RegisterMetrics from parsed .reg directives.

    Args:
        reg_directives: list of (type, name_prefix, count)
                        from parser.parse_reg_directives()

    Returns:
        RegisterMetrics object
    """
    metrics = RegisterMetrics()

    for reg_type, prefix, count in reg_directives:
        metrics.by_type[reg_type] = metrics.by_type.get(reg_type, 0) + count
        metrics.prefixes[reg_type] = prefix

    return metrics
