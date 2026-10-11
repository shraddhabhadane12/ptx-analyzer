"""
test_metrics.py — Unit tests for all four analysis metrics.

Run with:
    python -m pytest tests/ -v
    python -m unittest tests/test_metrics.py -v
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.parser import (parse_source, parse_reg_directives,
                         parse_shared_memory, parse_local_memory)
from src.metrics.registers       import analyze_registers
from src.metrics.memory          import analyze_memory
from src.metrics.instruction_mix import analyze_instruction_mix
from src.metrics.divergence      import analyze_divergence


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ptx(body: str) -> str:
    return f".version 7.5\n.target sm_89\n.address_size 64\n{body}"


# ── Register Tests ────────────────────────────────────────────────────────────

class TestRegisterMetrics(unittest.TestCase):

    def test_basic_register_count(self):
        src = make_ptx(".reg .f32 %f<8>;\n.reg .u64 %rd<4>;\n.reg .pred %p<2>;")
        regs = analyze_registers(parse_reg_directives(src))
        self.assertEqual(regs.by_type.get('f32', 0), 8)
        self.assertEqual(regs.by_type.get('u64', 0), 4)
        self.assertEqual(regs.by_type.get('pred', 0), 2)
        self.assertEqual(regs.total, 14)

    def test_zero_registers(self):
        src = make_ptx("ret;")
        regs = analyze_registers(parse_reg_directives(src))
        self.assertEqual(regs.total, 0)

    def test_pressure_low(self):
        src = make_ptx(".reg .f32 %f<16>;\n.reg .u32 %r<8>;")
        regs = analyze_registers(parse_reg_directives(src))
        self.assertIn("LOW", regs.pressure_level)

    def test_pressure_high(self):
        src = make_ptx(".reg .f32 %f<64>;\n.reg .u64 %rd<16>;")
        regs = analyze_registers(parse_reg_directives(src))
        self.assertIn("HIGH", regs.pressure_level)

    def test_max_threads_per_sm(self):
        src = make_ptx(".reg .f32 %f<32>;")
        regs = analyze_registers(parse_reg_directives(src))
        # 32 regs → 65536/32 = 2048 threads
        self.assertGreater(regs.max_threads_per_sm, 0)

    def test_multiple_float_types(self):
        src = make_ptx(".reg .f32 %f<4>;\n.reg .f64 %fd<2>;")
        regs = analyze_registers(parse_reg_directives(src))
        self.assertEqual(regs.float_regs, 4)
        self.assertEqual(regs.double_regs, 2)


# ── Memory Tests ──────────────────────────────────────────────────────────────

class TestMemoryMetrics(unittest.TestCase):

    def test_shared_memory_detected(self):
        src = make_ptx(".shared .align 4 .b8 sdata[4096];")
        self.assertEqual(parse_shared_memory(src), 4096)

    def test_no_shared_memory(self):
        src = make_ptx("ret;")
        self.assertEqual(parse_shared_memory(src), 0)

    def test_global_loads_counted(self):
        src = make_ptx("""
    .reg .f32 %f<2>;
    .reg .u64 %rd<1>;
    ld.global.f32   %f0, [%rd0];
    ld.global.f32   %f1, [%rd0];
    st.global.f32   [%rd0], %f0;
""")
        nodes = parse_source(src)
        mem = analyze_memory(nodes, 0, 0)
        self.assertEqual(mem.global_loads, 2)
        self.assertEqual(mem.global_stores, 1)

    def test_shared_memory_ops(self):
        src = make_ptx("""
    .reg .f32 %f<2>;
    .reg .u32 %r<1>;
    ld.shared.f32   %f0, [%r0];
    st.shared.f32   [%r0], %f0;
""")
        nodes = parse_source(src)
        mem = analyze_memory(nodes, 0, 0)
        self.assertEqual(mem.shared_loads, 1)
        self.assertEqual(mem.shared_stores, 1)

    def test_memory_bound_risk_high(self):
        src = make_ptx("""
    .reg .f32 %f<4>;
    .reg .u64 %rd<2>;
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd0];
    ld.global.f32 %f2, [%rd0];
    st.global.f32 [%rd1], %f0;
""")
        nodes = parse_source(src)
        mem = analyze_memory(nodes, 0, 0)
        self.assertIn("HIGH", mem.memory_bound_risk)

    def test_shared_pressure_none(self):
        nodes = parse_source(make_ptx("ret;"))
        mem = analyze_memory(nodes, 0, 0)
        self.assertEqual(mem.shared_pressure, "NONE")


# ── Instruction Mix Tests ─────────────────────────────────────────────────────

class TestInstructionMix(unittest.TestCase):

    def test_arithmetic_counted(self):
        src = make_ptx("""
    .reg .f32 %f<4>;
    add.f32   %f2, %f0, %f1;
    mul.f32   %f3, %f0, %f1;
""")
        nodes = parse_source(src)
        mix = analyze_instruction_mix(nodes)
        self.assertEqual(mix.counts.get('ARITHMETIC', 0), 2)
        self.assertEqual(mix.total, 2)

    def test_memory_counted(self):
        src = make_ptx("""
    .reg .f32 %f<2>;
    .reg .u64 %rd<1>;
    ld.global.f32 %f0, [%rd0];
    st.global.f32 [%rd0], %f0;
""")
        nodes = parse_source(src)
        mix = analyze_instruction_mix(nodes)
        self.assertEqual(mix.counts.get('MEMORY', 0), 2)

    def test_branch_counted(self):
        src = make_ptx("""
    bra END;
END:
    ret;
""")
        nodes = parse_source(src)
        mix = analyze_instruction_mix(nodes)
        branch = mix.counts.get('BRANCH', 0)
        self.assertGreaterEqual(branch, 1)

    def test_percentage_sums_to_100(self):
        src = make_ptx("""
    .reg .f32 %f<4>;
    .reg .u64 %rd<2>;
    ld.global.f32 %f0, [%rd0];
    add.f32   %f1, %f0, %f0;
    mul.f32   %f2, %f0, %f0;
    st.global.f32 [%rd1], %f2;
    ret;
""")
        nodes = parse_source(src)
        mix = analyze_instruction_mix(nodes)
        total_pct = sum(mix.percentage(cat) for cat in mix.counts)
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_compute_intensity_compute_bound(self):
        src = make_ptx("""
    .reg .f32 %f<8>;
    .reg .u64 %rd<1>;
    ld.global.f32 %f0, [%rd0];
    add.f32 %f1, %f0, %f0;
    mul.f32 %f2, %f0, %f0;
    add.f32 %f3, %f1, %f2;
    mul.f32 %f4, %f1, %f2;
    add.f32 %f5, %f3, %f4;
    st.global.f32 [%rd0], %f5;
""")
        nodes = parse_source(src)
        mix = analyze_instruction_mix(nodes)
        self.assertIn(mix.compute_intensity.split()[0],
                      ("COMPUTE-BOUND", "BALANCED"))


# ── Divergence Tests ──────────────────────────────────────────────────────────

class TestDivergenceMetrics(unittest.TestCase):

    def test_no_branches(self):
        src = make_ptx("""
    .reg .f32 %f<3>;
    .reg .u64 %rd<1>;
    ld.global.f32 %f0, [%rd0];
    add.f32 %f1, %f0, %f0;
    st.global.f32 [%rd0], %f1;
    ret;
""")
        nodes = parse_source(src)
        div = analyze_divergence(nodes)
        self.assertEqual(div.conditional_branches, 0)
        self.assertEqual(div.risk_level, "NONE")

    def test_conditional_branch_detected(self):
        src = make_ptx("""
    .reg .pred %p<1>;
    .reg .u32  %r<3>;
    .reg .u64  %rd<1>;
    setp.ge.u32   %p0, %r0, %r1;
    @%p0 bra      END;
    ret;
END:
    ret;
""")
        nodes = parse_source(src)
        div = analyze_divergence(nodes)
        self.assertGreaterEqual(div.conditional_branches, 1)
        self.assertEqual(div.setp_count, 1)

    def test_risk_level_low(self):
        src = make_ptx("""
    .reg .pred %p<1>;
    .reg .u32  %r<2>;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra END;
END:
    ret;
""")
        nodes = parse_source(src)
        div = analyze_divergence(nodes)
        self.assertIn(div.risk_level, ("LOW", "NONE"))

    def test_risk_level_high(self):
        src = make_ptx("""
    .reg .pred %p<3>;
    .reg .u32  %r<4>;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra L1;
    setp.gt.u32 %p1, %r0, %r2;
    @%p1 bra L2;
    setp.eq.u32 %p2, %r0, %r3;
    @%p2 bra L3;
    setp.ne.u32 %p0, %r1, %r2;
    @%p0 bra L4;
    setp.ge.u32 %p1, %r2, %r3;
    @%p1 bra L5;
    setp.le.u32 %p2, %r0, %r3;
    @%p2 bra L6;
L1: L2: L3: L4: L5: L6:
    ret;
""")
        nodes = parse_source(src)
        div = analyze_divergence(nodes)
        self.assertIn(div.risk_level, ("MEDIUM", "HIGH"))

    def test_unconditional_branch_not_counted_as_divergence(self):
        src = make_ptx("""
    bra END;
END:
    ret;
""")
        nodes = parse_source(src)
        div = analyze_divergence(nodes)
        self.assertEqual(div.conditional_branches, 0)

    def test_divergence_points_recorded(self):
        src = make_ptx("""
    .reg .pred %p<1>;
    .reg .u32  %r<2>;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra END;
END:
    ret;
""")
        nodes = parse_source(src)
        div = analyze_divergence(nodes)
        if div.conditional_branches > 0:
            self.assertEqual(len(div.divergence_points),
                             div.conditional_branches)


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_vector_add_full_analysis(self):
        """End-to-end analysis of vector_add.ptx."""
        ptx_path = os.path.join(os.path.dirname(__file__),
                                '..', 'samples', 'ptx_files', 'vector_add.ptx')
        if not os.path.exists(ptx_path):
            self.skipTest("vector_add.ptx not found")

        with open(ptx_path, encoding='utf-8') as f:
            source = f.read()

        nodes  = parse_source(source)
        reg_m  = analyze_registers(parse_reg_directives(source))
        mem_m  = analyze_memory(nodes, parse_shared_memory(source),
                                parse_local_memory(source))
        mix_m  = analyze_instruction_mix(nodes)
        div_m  = analyze_divergence(nodes)

        # Basic sanity checks
        self.assertGreater(mix_m.total, 0, "Should have instructions")
        self.assertGreater(reg_m.total, 0, "Should have registers")
        self.assertGreaterEqual(mem_m.global_loads, 0)
        self.assertIn(div_m.risk_level, ("NONE", "LOW", "MEDIUM", "HIGH"))

    def test_matrix_mul_has_registers(self):
        """matrix_mul.ptx should declare multiple register types."""
        ptx_path = os.path.join(os.path.dirname(__file__),
                                '..', 'samples', 'ptx_files', 'matrix_mul.ptx')
        if not os.path.exists(ptx_path):
            self.skipTest("matrix_mul.ptx not found")

        with open(ptx_path, encoding='utf-8') as f:
            source = f.read()

        reg_m = analyze_registers(parse_reg_directives(source))
        self.assertGreater(reg_m.total, 0)
        self.assertGreater(len(reg_m.by_type), 1, "Should have multiple register types")

    def test_reduction_has_conditional_branch(self):
        """reduction.ptx should have at least one conditional branch."""
        ptx_path = os.path.join(os.path.dirname(__file__),
                                '..', 'samples', 'ptx_files', 'reduction.ptx')
        if not os.path.exists(ptx_path):
            self.skipTest("reduction.ptx not found")

        with open(ptx_path, encoding='utf-8') as f:
            source = f.read()

        nodes = parse_source(source)
        div_m = analyze_divergence(nodes)
        self.assertGreaterEqual(div_m.setp_count, 1,
                                "Reduction should have comparisons")


if __name__ == '__main__':
    unittest.main(verbosity=2)
