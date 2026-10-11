# PTX Analyzer

A static analysis tool for NVIDIA PTX (Parallel Thread Execution) assembly that reports **register usage**, **memory footprint**, **instruction mix**, and **branch divergence risk** for GPU kernels.

Built as a compiler engineering portfolio project targeting the **NVIDIA PTX Compiler Intern** role — companion tool to [ptx-optimizer](https://github.com/shraddhabhadane12/ptx-optimizer).

```
CUDA C++ → PTX (nvcc) → [THIS TOOL] → Analysis Report
```

---

## What It Analyzes

### Register Usage
```
Float  32-bit (%f)  : 8
Addr   64-bit (%rd) : 4
Int    32-bit (%r)  : 5
Predicate (%p)      : 2
TOTAL               : 19
Pressure            : LOW
Est. max threads/SM : 2048
```
More registers per thread = fewer threads can run simultaneously = lower GPU occupancy.

### Memory Usage
```
Shared memory declared: 0 bytes  (0.0 KB)
Global loads          : 2
Global stores         : 1
Memory-bound risk     : LOW (mostly shared/const memory)
```
Kernels with many `ld.global` accesses are memory-bound — a key performance bottleneck.

### Instruction Mix
```
ARITHMETIC   :   5  (20.0%)  ████░░░░░░░░░░░░
MEMORY       :   3  (12.0%)  ██░░░░░░░░░░░░░░
MOVEMENT     :   8  (32.0%)  ██████░░░░░░░░░░
COMPARISON   :   2  ( 8.0%)  █░░░░░░░░░░░░░░░
BRANCH       :   3  (12.0%)  ██░░░░░░░░░░░░░░
Compute intensity: COMPUTE-BOUND (ratio 1.7:1)
```

### Branch Divergence
```
setp instructions     : 2
Conditional branches  : 2
Risk level            : LOW
Recommendation        : Few conditional branches — likely low impact.
```
GPU threads run in warps of 32. When threads take different branches, the warp serializes → 2x slower. Your tool detects every `setp + @%p bra` pair.

---

## Project Structure

```
ptx-analyzer/
├── src/
│   ├── ir.py                     # Instruction / RawLine data structures
│   ├── parser.py                 # PTX parser + directive extractors
│   ├── reporter.py               # Report formatter (table + JSON)
│   ├── analyzer.py               # Main driver + CLI
│   └── metrics/
│       ├── registers.py          # Register usage + pressure analysis
│       ├── memory.py             # Shared/global memory analysis
│       ├── instruction_mix.py    # Instruction categorization
│       └── divergence.py         # Branch divergence detection
├── samples/
│   └── ptx_files/                # Sample PTX files (no CUDA needed)
│       ├── vector_add.ptx
│       ├── matrix_mul.ptx
│       └── reduction.ptx
├── tests/
│   └── test_metrics.py           # 20 unit tests
└── README.md
```

---

## Quick Start

**Requirements:** Python 3.10+, no external dependencies.

```bash
# Clone
git clone https://github.com/shraddhabhadane12/ptx-analyzer.git
cd ptx-analyzer

# Analyze a single PTX file
python -m src.analyzer samples/ptx_files/vector_add.ptx

# Verbose output (shows divergence point locations)
python -m src.analyzer samples/ptx_files/vector_add.ptx --verbose

# JSON output (for scripting / CI pipelines)
python -m src.analyzer samples/ptx_files/vector_add.ptx --json

# Compare all PTX files side by side
python -m src.analyzer samples/ptx_files/

# Run all unit tests
python -m unittest tests/test_metrics.py -v
```

---

## Sample Output

```
╔════════════════════════════════════════════════════════════╗
║                    PTX ANALYSIS REPORT                    ║
║                 File: vector_add.ptx                      ║
╠════════════════════════════════════════════════════════════╣
║  REGISTER USAGE                                           ║
║  ──────────────────────────────────────────              ║
║    Float  32-bit (%f)  : 16                              ║
║    Addr   64-bit (%rd) : 8                               ║
║    Int    32-bit (%r)  : 5                               ║
║    Predicate (%p)      : 3                               ║
║                                                          ║
║    TOTAL registers       : 32                            ║
║    Pressure              : LOW                           ║
║    Est. max threads/SM   : 2048                          ║
╠════════════════════════════════════════════════════════════╣
║  MEMORY USAGE                                            ║
║    Shared memory declared: 0 bytes  (0.0 KB)             ║
║    Global loads          : 2                             ║
║    Global stores         : 1                             ║
║    Memory-bound risk     : LOW                           ║
╠════════════════════════════════════════════════════════════╣
║  INSTRUCTION MIX  (25 total instructions)                ║
║    ARITHMETIC   :   5  (20.0%)  ████░░░░░░░░░░░░        ║
║    MEMORY       :   3  (12.0%)  ██░░░░░░░░░░░░░░        ║
║    MOVEMENT     :   8  (32.0%)  ██████░░░░░░░░░░        ║
║    COMPARISON   :   2  ( 8.0%)  █░░░░░░░░░░░░░░░        ║
║    BRANCH       :   3  (12.0%)  ██░░░░░░░░░░░░░░        ║
║    Compute intensity: BALANCED (ratio 1.7:1)             ║
╠════════════════════════════════════════════════════════════╣
║  BRANCH DIVERGENCE ANALYSIS                              ║
║    setp instructions     : 2                             ║
║    Conditional branches  : 2                             ║
║    Risk level            : LOW                           ║
║    Recommendation: Few conditional branches —            ║
║    likely low impact on performance.                     ║
╚════════════════════════════════════════════════════════════╝
```

---

## Compare Mode

```bash
python -m src.analyzer samples/ptx_files/
```

```
  File                           Regs  Shared    Instrs  Diverge   Bound
  ----------------------------------------------------------------------
  vector_add.ptx                   19       0B      25  LOW       LOW
  matrix_mul.ptx                   26       0B      31  LOW       LOW
  reduction.ptx                    22       0B      28  LOW       MEDIUM
```

---

## JSON Output

```bash
python -m src.analyzer samples/ptx_files/vector_add.ptx --json
```

```json
{
  "file": "vector_add.ptx",
  "registers": {
    "by_type": {"f32": 16, "u64": 8, "u32": 5, "pred": 3},
    "total": 32,
    "pressure_level": "LOW",
    "max_threads_per_sm": 2048
  },
  "memory": {
    "shared_bytes": 0,
    "global_loads": 2,
    "global_stores": 1,
    "memory_bound_risk": "LOW (mostly shared/const memory)"
  },
  "instruction_mix": {
    "total": 25,
    "counts": {"ARITHMETIC": 5, "MEMORY": 3, ...}
  },
  "divergence": {
    "conditional_branches": 2,
    "risk_level": "LOW"
  }
}
```

---

## Skills Demonstrated

| Concept | Implementation |
|---|---|
| **PTX ISA** | Parses `.reg`, `.shared`, `setp`, `bra`, `ld.global` |
| **GPU Architecture** | Register pressure → occupancy, warp divergence |
| **Static Analysis** | Read-only code analysis without execution |
| **CUDA Toolchain** | Can auto-invoke `nvcc --ptx` for `.cu` files |
| **Memory Hierarchy** | Global vs shared vs local memory classification |
| **Branch Analysis** | Detects `setp + @%p bra` divergence patterns |
| **Data Reporting** | ASCII bar charts, JSON output, compare mode |

---

## Related Project

[ptx-optimizer](https://github.com/shraddhabhadane12/ptx-optimizer) — Takes PTX assembly and removes dead instructions, folds constants, and applies algebraic simplifications to reduce instruction count by 50%+.

Together these tools form a PTX analysis + optimization pipeline:
```
PTX Analyzer  →  identify bottlenecks  →  PTX Optimizer  →  remove waste
```

---

## References

- [NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Understanding Warp Divergence](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

---

## License

MIT License
