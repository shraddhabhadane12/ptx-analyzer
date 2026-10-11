"""
analyzer.py — Main Driver & CLI

Entry point for the PTX Analyzer tool.

Usage:
    python -m src.analyzer input.ptx [options]
    python -m src.analyzer samples/ptx_files/vector_add.ptx --verbose
    python -m src.analyzer samples/ptx_files/ --all          (analyze directory)
    python -m src.analyzer input.ptx --json                   (JSON output)
    python -m src.analyzer a.ptx b.ptx --compare             (side-by-side)

If nvcc is available, can also accept .cu files:
    python -m src.analyzer kernel.cu                          (auto-compiles)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from .parser import parse_source, parse_reg_directives, parse_shared_memory, parse_local_memory
from .metrics.registers       import analyze_registers
from .metrics.memory          import analyze_memory
from .metrics.instruction_mix import analyze_instruction_mix
from .metrics.divergence      import analyze_divergence
from .reporter import print_report, print_json, print_summary


# ── Core analysis pipeline ────────────────────────────────────────────────────

def analyze_ptx(source: str, filename: str = "unknown",
                verbose: bool = False, as_json: bool = False) -> dict:
    """
    Run the full analysis pipeline on PTX source text.

    Args:
        source:   PTX file contents as string
        filename: display name for the report header
        verbose:  show extra detail (divergence points, etc.)
        as_json:  output JSON instead of table

    Returns:
        dict with all metric objects and raw stats
    """
    # Parse into IR nodes
    nodes = parse_source(source)

    # Extract metadata from directives
    reg_directives = parse_reg_directives(source)
    shared_bytes   = parse_shared_memory(source)
    local_bytes    = parse_local_memory(source)

    # Run all four metrics
    reg_m = analyze_registers(reg_directives)
    mem_m = analyze_memory(nodes, shared_bytes, local_bytes)
    mix_m = analyze_instruction_mix(nodes)
    div_m = analyze_divergence(nodes)

    if as_json:
        print_json(filename, reg_m, mem_m, mix_m, div_m)
    else:
        print_report(filename, reg_m, mem_m, mix_m, div_m, verbose=verbose)

    return {
        'registers':        reg_m,
        'memory':           mem_m,
        'instruction_mix':  mix_m,
        'divergence':       div_m,
    }


def _compile_cuda_to_ptx(cu_path: Path) -> Optional[str]:
    """
    Compile a .cu file to PTX using nvcc.
    Returns PTX source string or None if nvcc is not available.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.ptx', delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            ['nvcc', '--ptx', str(cu_path), '-o', tmp_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"nvcc error:\n{result.stderr}", file=sys.stderr)
            return None

        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()

    except FileNotFoundError:
        print("nvcc not found. Install CUDA toolkit or provide a .ptx file directly.",
              file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("nvcc timed out.", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _analyze_file(path: Path, verbose: bool, as_json: bool) -> Optional[dict]:
    """Analyze a single .ptx or .cu file."""
    if path.suffix.lower() == '.cu':
        print(f"Compiling {path.name} with nvcc...", file=sys.stderr)
        source = _compile_cuda_to_ptx(path)
        if source is None:
            return None
    else:
        try:
            source = path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {path}: {e}", file=sys.stderr)
            return None

    return analyze_ptx(source, filename=path.name, verbose=verbose, as_json=as_json)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='ptx-analyzer',
        description='PTX Analyzer — Static analysis of NVIDIA PTX assembly kernels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.analyzer samples/ptx_files/vector_add.ptx
  python -m src.analyzer samples/ptx_files/vector_add.ptx --verbose
  python -m src.analyzer samples/ptx_files/vector_add.ptx --json
  python -m src.analyzer samples/ptx_files/            (analyze all .ptx files)
  python -m src.analyzer a.ptx b.ptx c.ptx --compare
        """
    )
    p.add_argument('inputs', nargs='+',
                   help='PTX file(s), .cu file(s), or a directory of PTX files')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Show extra detail (divergence point locations, etc.)')
    p.add_argument('--json', action='store_true',
                   help='Output results as JSON')
    p.add_argument('--compare', action='store_true',
                   help='Print one-line summary per file for easy comparison')
    p.add_argument('--only',
                   choices=['registers', 'memory', 'mix', 'divergence'],
                   help='Show only one metric section')
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Expand directories to PTX files
    paths: List[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            ptx_files = sorted(p.glob('*.ptx'))
            if not ptx_files:
                print(f"No .ptx files found in {p}", file=sys.stderr)
            paths.extend(ptx_files)
        elif p.exists():
            paths.append(p)
        else:
            print(f"File not found: {p}", file=sys.stderr)
            return 1

    if not paths:
        print("No input files to analyze.", file=sys.stderr)
        return 1

    # Compare mode: one-line summary per file
    if args.compare or len(paths) > 1:
        print()
        print(f"  {'File':<30} {'Regs':>4}  {'Shared':>8}  "
              f"{'Instrs':>6}  {'Diverge':<8}  Bound")
        print("  " + "-" * 70)

        for path in paths:
            source = path.read_text(encoding='utf-8')
            nodes  = parse_source(source)
            reg_m  = analyze_registers(parse_reg_directives(source))
            mem_m  = analyze_memory(nodes, parse_shared_memory(source),
                                    parse_local_memory(source))
            mix_m  = analyze_instruction_mix(nodes)
            div_m  = analyze_divergence(nodes)
            print_summary(path.name, reg_m, mem_m, mix_m, div_m)

        print()
        return 0

    # Single file: full report
    for path in paths:
        result = _analyze_file(path, verbose=args.verbose, as_json=args.json)
        if result is None:
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
