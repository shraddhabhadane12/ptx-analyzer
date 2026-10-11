"""
ir.py — Intermediate Representation for PTX analysis.

Two node types:
  Instruction — a parsed executable PTX line
  RawLine     — directives, labels, comments (passed through)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Instruction:
    """A single executable PTX instruction."""
    opcode:   str                    # e.g. "add.f32", "ld.global.f32"
    dest:     Optional[str]          # destination register or None
    operands: List[str]              # source operands
    raw:      str                    # original text line
    lineno:   int = 0                # line number in file (1-based)

    def base_op(self) -> str:
        """Return base opcode without type suffix: 'add.f32' → 'add'"""
        return self.opcode.split('.')[0]

    def is_store(self)   -> bool: return self.opcode.startswith('st.')
    def is_load(self)    -> bool: return self.opcode.startswith('ld.')
    def is_mov(self)     -> bool: return self.opcode.startswith('mov.')
    def is_branch(self)  -> bool:
        return self.base_op() in ('bra', 'ret', 'exit', 'call', 'brx')
    def is_barrier(self) -> bool:
        return self.base_op() in ('bar', 'membar')
    def is_predicated(self) -> bool:
        """True if line starts with @%p — a conditional predicated instruction."""
        return self.raw.strip().startswith('@')


@dataclass
class RawLine:
    """A non-instruction PTX line (directive, label, blank, comment)."""
    text:   str
    lineno: int = 0


# Type alias used throughout
PTXNode = Instruction | RawLine
