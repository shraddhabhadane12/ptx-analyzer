"""
metrics/memory.py — Memory Usage Analysis

Analyzes PTX for:
  1. Shared memory declared (.shared directives)
  2. Local memory declared (.local directives)
  3. Global memory access patterns (ld.global / st.global counts)
  4. Shared memory access patterns (ld.shared / st.shared counts)
  5. Memory access type breakdown

KEY CONCEPT — GPU Memory Hierarchy (fastest to slowest):
    Registers    ~1 cycle      (private per thread)
    Shared mem   ~4 cycles     (shared within thread block) ← fast!
    L1 cache     ~30 cycles    (automatic)
    L2 cache     ~100 cycles   (automatic)
    Global mem   ~200-600 cycles (DRAM — very slow!)

A kernel doing many ld.global accesses may be memory-bound.
A kernel using shared memory well is typically faster.

SHARED MEMORY LIMIT:
    Most NVIDIA GPUs: 48KB or 96KB per block
    Using too much → fewer blocks can run simultaneously → lower occupancy
"""

from dataclasses import dataclass


@dataclass
class MemoryMetrics:
    """Complete memory usage metrics for one PTX kernel."""

    # Declared memory sizes
    shared_bytes: int = 0     # .shared [...] total
    local_bytes:  int = 0     # .local  [...] total (thread-private stack)

    # Instruction counts by memory space
    global_loads:  int = 0    # ld.global.*
    global_stores: int = 0    # st.global.*
    shared_loads:  int = 0    # ld.shared.*
    shared_stores: int = 0    # st.shared.*
    local_loads:   int = 0    # ld.local.*
    local_stores:  int = 0    # st.local.*
    const_loads:   int = 0    # ld.const.*
    param_loads:   int = 0    # ld.param.*

    @property
    def total_loads(self) -> int:
        return (self.global_loads + self.shared_loads +
                self.local_loads + self.const_loads + self.param_loads)

    @property
    def total_stores(self) -> int:
        return self.global_stores + self.shared_stores + self.local_stores

    @property
    def total_memory_ops(self) -> int:
        return self.total_loads + self.total_stores

    @property
    def shared_memory_kb(self) -> float:
        return self.shared_bytes / 1024.0

    @property
    def shared_pressure(self) -> str:
        """Classify shared memory usage relative to typical 48KB limit."""
        kb = self.shared_memory_kb
        if kb == 0:    return "NONE"
        if kb < 8:     return "LOW"
        if kb < 24:    return "MEDIUM"
        if kb < 40:    return "HIGH"
        return "VERY HIGH (near 48KB limit)"

    @property
    def memory_bound_risk(self) -> str:
        """
        Estimate if the kernel is likely memory-bound.
        High ratio of global memory accesses = memory-bound = potential bottleneck.
        """
        if self.total_memory_ops == 0:
            return "N/A"
        global_ratio = (self.global_loads + self.global_stores) / self.total_memory_ops
        if global_ratio > 0.7:
            return "HIGH (many global memory accesses)"
        if global_ratio > 0.4:
            return "MEDIUM"
        return "LOW (mostly shared/const memory)"


def analyze_memory(nodes, shared_bytes: int, local_bytes: int) -> MemoryMetrics:
    """
    Build MemoryMetrics by scanning instruction nodes + declared sizes.

    Args:
        nodes:        parsed PTXNode list
        shared_bytes: from parser.parse_shared_memory()
        local_bytes:  from parser.parse_local_memory()

    Returns:
        MemoryMetrics object
    """
    from ..ir import Instruction

    metrics = MemoryMetrics(
        shared_bytes=shared_bytes,
        local_bytes=local_bytes,
    )

    for node in nodes:
        if not isinstance(node, Instruction):
            continue

        op = node.opcode.lower()

        if op.startswith('ld.global'):   metrics.global_loads  += 1
        elif op.startswith('st.global'): metrics.global_stores += 1
        elif op.startswith('ld.shared'): metrics.shared_loads  += 1
        elif op.startswith('st.shared'): metrics.shared_stores += 1
        elif op.startswith('ld.local'):  metrics.local_loads   += 1
        elif op.startswith('st.local'):  metrics.local_stores  += 1
        elif op.startswith('ld.const'):  metrics.const_loads   += 1
        elif op.startswith('ld.param'):  metrics.param_loads   += 1

    return metrics
