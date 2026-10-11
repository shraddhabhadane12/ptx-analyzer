"""
parser.py — PTX Parser for the Analyzer.

Parses PTX source text into a list of PTXNodes (Instruction | RawLine).
Also exposes helper functions to parse .reg and .shared directives
which carry register and memory metadata needed by the metrics.
"""

import re
from typing import List, Tuple, Optional

from .ir import Instruction, RawLine, PTXNode


# ── Patterns ──────────────────────────────────────────────────────────────────

# Instruction line: must be indented, starts with opcode
_RE_INSTR = re.compile(
    r'^\s*(?:@(!?)(%\w+)\s+)?'    # optional predicate
    r'([\w][\w\.]*)'               # opcode
    r'(?:\s+(.+?))?'               # operands (lazy)
    r'\s*;?\s*(?://.*)?$'
)

# .reg directive: .reg .TYPE %name<N>;
_RE_REG = re.compile(
    r'^\s*\.reg\s+\.(b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f16|f32|f64|pred)\s+(%\w+)<(\d+)>'
)

# .shared directive: .shared .align N .bX name[SIZE];
_RE_SHARED = re.compile(
    r'^\s*\.shared\b.*\[(\d+)\]'
)

# .param directive: .param .TYPE name[SIZE] — for kernel parameter shared mem
_RE_PARAM_ARRAY = re.compile(
    r'^\s*\.local\b.*\[(\d+)\]'
)

_RE_REGISTER = re.compile(r'^%\w+$')

NO_DEST_PREFIXES = (
    'st.', 'ret', 'bra', 'exit', 'bar', 'call',
    'trap', 'membar', 'prefetch', 'prefetchu',
)


def _has_dest(opcode: str) -> bool:
    for p in NO_DEST_PREFIXES:
        if opcode == p.rstrip('.') or opcode.startswith(p):
            return False
    return True


def _split_operands(s: str) -> List[str]:
    """Split operand string on commas, respecting brackets."""
    s = re.sub(r'//.*$', '', s).rstrip(';').strip()
    if not s:
        return []
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == '[':   depth += 1; cur.append(ch)
        elif ch == ']': depth -= 1; cur.append(ch)
        elif ch == ',' and depth == 0:
            p = ''.join(cur).strip()
            if p: parts.append(p)
            cur = []
        else:
            cur.append(ch)
    p = ''.join(cur).strip()
    if p: parts.append(p)
    return parts


# ── Public parse functions ────────────────────────────────────────────────────

def parse_source(source: str) -> List[PTXNode]:
    """
    Parse full PTX source text into a list of PTXNodes.

    Args:
        source: PTX file contents as string

    Returns:
        list of Instruction and RawLine objects
    """
    nodes: List[PTXNode] = []

    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()

        # Skip blank / comment / directive lines as RawLine
        if not stripped or stripped.startswith('//') or stripped.startswith('.') \
                or (stripped and stripped[0] not in ('@', '%')
                    and not (line and (line[0] == ' ' or line[0] == '\t'))):
            nodes.append(RawLine(text=line, lineno=lineno))
            continue

        m = _RE_INSTR.match(line)
        if not m:
            nodes.append(RawLine(text=line, lineno=lineno))
            continue

        pred_neg  = m.group(1) or ''
        pred_reg  = m.group(2) or ''
        opcode    = m.group(3)
        op_str    = m.group(4) or ''

        operands = _split_operands(op_str)

        dest = None
        src_ops = operands
        if _has_dest(opcode) and operands and _RE_REGISTER.match(operands[0]):
            dest    = operands[0]
            src_ops = operands[1:]

        nodes.append(Instruction(
            opcode=opcode,
            dest=dest,
            operands=src_ops,
            raw=line,
            lineno=lineno,
        ))

    return nodes


def parse_reg_directives(source: str) -> List[Tuple[str, str, int]]:
    """
    Extract register declarations from PTX source.

    Returns:
        list of (type, name_prefix, count)
        e.g. [("f32", "%f", 8), ("u64", "%rd", 4), ("pred", "%p", 2)]
    """
    regs = []
    for line in source.splitlines():
        m = _RE_REG.match(line)
        if m:
            reg_type   = m.group(1)
            name_prefix = m.group(2)
            count       = int(m.group(3))
            regs.append((reg_type, name_prefix, count))
    return regs


def parse_shared_memory(source: str) -> int:
    """
    Return total shared memory declared (bytes) in the PTX source.
    Sums all .shared [...] declarations.
    """
    total = 0
    for line in source.splitlines():
        m = _RE_SHARED.match(line)
        if m:
            total += int(m.group(1))
    return total


def parse_local_memory(source: str) -> int:
    """Return total .local memory declared (bytes)."""
    total = 0
    for line in source.splitlines():
        m = _RE_PARAM_ARRAY.match(line)
        if m:
            total += int(m.group(1))
    return total
