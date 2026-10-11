"""
reporter.py — Report Formatter

Takes all four metric objects and renders a clean, readable
analysis report to stdout. Supports both terminal (ASCII art)
and JSON output modes.
"""

import json
from typing import Optional
from .metrics.registers      import RegisterMetrics
from .metrics.memory         import MemoryMetrics
from .metrics.instruction_mix import InstructionMixMetrics
from .metrics.divergence     import DivergenceMetrics


WIDTH = 62   # report box width


def _line(text: str = "") -> str:
    return f"║  {text:<{WIDTH - 4}}║"


def _divider() -> str:
    return "╠" + "═" * (WIDTH - 2) + "╣"


def _top() -> str:
    return "╔" + "═" * (WIDTH - 2) + "╗"


def _bottom() -> str:
    return "╚" + "═" * (WIDTH - 2) + "╝"


def _header(title: str) -> str:
    pad = (WIDTH - 2 - len(title)) // 2
    return "║" + " " * pad + title + " " * (WIDTH - 2 - pad - len(title)) + "║"


def print_report(
    filename: str,
    reg_m:   RegisterMetrics,
    mem_m:   MemoryMetrics,
    mix_m:   InstructionMixMetrics,
    div_m:   DivergenceMetrics,
    verbose: bool = False,
) -> None:
    """Print the full analysis report to stdout."""

    lines = []
    lines.append(_top())
    lines.append(_header("PTX ANALYSIS REPORT"))
    lines.append(_header(f"File: {filename}"))
    lines.append(_divider())

    # ── Register Usage ────────────────────────────────────────────────────
    lines.append(_line("REGISTER USAGE"))
    lines.append(_line("─" * (WIDTH - 6)))

    type_labels = {
        'f32':  'Float  32-bit (%f) ',
        'f64':  'Float  64-bit (%fd)',
        'f16':  'Float  16-bit (%h) ',
        'u32':  'Int    32-bit (%r) ',
        'u64':  'Addr   64-bit (%rd)',
        's32':  'Int    32-bit (%r) ',
        's64':  'Int    64-bit     ',
        'b32':  'Bits   32-bit (%b) ',
        'b64':  'Bits   64-bit     ',
        'pred': 'Predicate (%p)     ',
    }

    for reg_type, count in sorted(reg_m.by_type.items()):
        label = type_labels.get(reg_type, f'{reg_type:<18}')
        prefix = reg_m.prefixes.get(reg_type, '')
        lines.append(_line(f"  {label} : {count}"))

    lines.append(_line())
    lines.append(_line(f"  TOTAL registers       : {reg_m.total}"))
    lines.append(_line(f"  Pressure              : {reg_m.pressure_level}"))
    lines.append(_line(f"  Est. max threads/SM   : {reg_m.max_threads_per_sm}"))

    # ── Memory ────────────────────────────────────────────────────────────
    lines.append(_divider())
    lines.append(_line("MEMORY USAGE"))
    lines.append(_line("─" * (WIDTH - 6)))
    lines.append(_line(f"  Shared memory declared: {mem_m.shared_bytes} bytes"
                        f"  ({mem_m.shared_memory_kb:.1f} KB)"))
    lines.append(_line(f"  Local  memory declared: {mem_m.local_bytes} bytes"))
    lines.append(_line(f"  Shared mem pressure   : {mem_m.shared_pressure}"))
    lines.append(_line())
    lines.append(_line(f"  Global loads          : {mem_m.global_loads}"))
    lines.append(_line(f"  Global stores         : {mem_m.global_stores}"))
    lines.append(_line(f"  Shared loads          : {mem_m.shared_loads}"))
    lines.append(_line(f"  Shared stores         : {mem_m.shared_stores}"))
    lines.append(_line(f"  Const/Param loads     : {mem_m.const_loads + mem_m.param_loads}"))
    lines.append(_line())
    lines.append(_line(f"  Memory-bound risk     : {mem_m.memory_bound_risk}"))

    # ── Instruction Mix ───────────────────────────────────────────────────
    lines.append(_divider())
    lines.append(_line(f"INSTRUCTION MIX  ({mix_m.total} total instructions)"))
    lines.append(_line("─" * (WIDTH - 6)))

    cat_order = ['ARITHMETIC', 'MEMORY', 'MOVEMENT', 'COMPARISON',
                 'BRANCH', 'SPECIAL', 'OTHER']
    for cat in cat_order:
        count = mix_m.counts.get(cat, 0)
        pct   = mix_m.percentage(cat)
        bar   = mix_m.bar_chart(cat, width=16)
        lines.append(_line(f"  {cat:<12} : {count:>3}  ({pct:>5.1f}%)  {bar}"))

    lines.append(_line())
    lines.append(_line(f"  Dominant category     : {mix_m.dominant_category}"))
    lines.append(_line(f"  Compute intensity     : {mix_m.compute_intensity}"))

    # ── Branch Divergence ─────────────────────────────────────────────────
    lines.append(_divider())
    lines.append(_line("BRANCH DIVERGENCE ANALYSIS"))
    lines.append(_line("─" * (WIDTH - 6)))
    lines.append(_line(f"  setp instructions     : {div_m.setp_count}"))
    lines.append(_line(f"  Conditional branches  : {div_m.conditional_branches}"))
    lines.append(_line(f"  Unconditional branches: {div_m.unconditional_branches}"))
    lines.append(_line(f"  Predicated instrs     : {div_m.predicated_instructions}"))
    lines.append(_line())
    lines.append(_line(f"  Risk level  {div_m.risk_color_hint}         : {div_m.risk_level}"))

    if verbose and div_m.divergence_points:
        lines.append(_line())
        lines.append(_line("  Divergence points:"))
        for dp in div_m.divergence_points[:5]:   # show max 5
            lines.append(_line(f"    Line {dp.lineno}: {dp.branch_raw[:40]}"))
            if dp.setp_raw:
                lines.append(_line(f"      ← {dp.setp_raw[:40]}"))

    lines.append(_line())
    lines.append(_line(f"  Recommendation:"))
    # Word-wrap recommendation across multiple lines
    rec = div_m.recommendation
    words = rec.split()
    cur_line = "    "
    for word in words:
        if len(cur_line) + len(word) + 1 > WIDTH - 6:
            lines.append(_line(cur_line))
            cur_line = "    " + word
        else:
            cur_line += (" " if cur_line != "    " else "") + word
    if cur_line.strip():
        lines.append(_line(cur_line))

    lines.append(_bottom())

    print('\n'.join(lines))


def print_json(
    filename: str,
    reg_m:   RegisterMetrics,
    mem_m:   MemoryMetrics,
    mix_m:   InstructionMixMetrics,
    div_m:   DivergenceMetrics,
) -> None:
    """Output results as JSON (useful for scripting / CI pipelines)."""
    data = {
        "file": filename,
        "registers": {
            "by_type":          reg_m.by_type,
            "total":            reg_m.total,
            "pressure_level":   reg_m.pressure_level,
            "max_threads_per_sm": reg_m.max_threads_per_sm,
        },
        "memory": {
            "shared_bytes":       mem_m.shared_bytes,
            "local_bytes":        mem_m.local_bytes,
            "global_loads":       mem_m.global_loads,
            "global_stores":      mem_m.global_stores,
            "shared_loads":       mem_m.shared_loads,
            "shared_stores":      mem_m.shared_stores,
            "memory_bound_risk":  mem_m.memory_bound_risk,
        },
        "instruction_mix": {
            "total":     mix_m.total,
            "counts":    mix_m.counts,
            "percentages": {k: round(mix_m.percentage(k), 2)
                            for k in mix_m.counts},
            "compute_intensity": mix_m.compute_intensity,
        },
        "divergence": {
            "setp_count":             div_m.setp_count,
            "conditional_branches":   div_m.conditional_branches,
            "unconditional_branches": div_m.unconditional_branches,
            "predicated_instructions":div_m.predicated_instructions,
            "risk_level":             div_m.risk_level,
            "recommendation":         div_m.recommendation,
        },
    }
    print(json.dumps(data, indent=2))


def print_summary(
    filename: str,
    reg_m:   RegisterMetrics,
    mem_m:   MemoryMetrics,
    mix_m:   InstructionMixMetrics,
    div_m:   DivergenceMetrics,
) -> None:
    """One-line summary for use when analyzing multiple files."""
    print(
        f"  {filename:<30} "
        f"regs={reg_m.total:>3}  "
        f"shared={mem_m.shared_bytes:>6}B  "
        f"instrs={mix_m.total:>4}  "
        f"diverge={div_m.risk_level:<6}  "
        f"bound={mem_m.memory_bound_risk.split()[0]}"
    )
