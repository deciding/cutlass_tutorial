"""
test_clock_smem.py
==================
CuTe DSL kernel that:
  1. Reads the HW clock register via inline PTX  (mov.u32 %r, %clock;)
  2. Saves it to shared memory                   (st.shared.u32 [smem], %r;)
  3. Reads it back from shared memory            (ld.shared.u32 %r, [smem];)
  4. Stores the result to global memory          (st.global.u32 [gmem], %r;)

Run:
    python test_clock_smem.py

Requirements:
    pip install nvidia-cutlass-dsl torch
    GPU with compute capability >= 7.0
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, arith
from cutlass._mlir import ir
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# 1.  Shared-memory layout struct
#     One u32 slot per thread (NUM_THREADS threads per block).
# ---------------------------------------------------------------------------
NUM_THREADS = 128   # change freely; keep ≤ 1024


@cute.struct
class SharedStorage:
    # MemRange[dtype, count]  — allocates count * sizeof(dtype) bytes
    clock_buf: cute.struct.MemRange[cutlass.Uint64, NUM_THREADS]


# ---------------------------------------------------------------------------
# 2.  The kernel
# ---------------------------------------------------------------------------
@cute.kernel
def clock_kernel(out: cute.Tensor):
    """
    Each thread:
      - reads its own thread index (tidx)
      - reads the SM clock register
      - stores clock → smem[tidx]
      - loads  smem[tidx] → register
      - stores register → gmem out[tidx]
    """
    tidx, _, _ = cute.arch.thread_idx()

    # ---- allocate smem -------------------------------------------------
    smem    = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # storage.clock_buf is a cute.Pointer  (addrspace 3, dtype Uint32)
    # .data_ptr() gives the base cute.Pointer of that field
    clock_ptr = storage.clock_buf.data_ptr()

    # Extract the raw !llvm.ptr<3> ir.Value
    # (attribute name is `.ir_value` in CUTLASS 4.x)
    smem_base_llvm = clock_ptr.llvm_ptr          # !llvm.ptr<3>

    # Cast smem pointer → i32  (PTX shared-memory addresses are 32-bit)
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)

    #smem_base_i32 = llvm.ptrtoint(i32, smem_base_llvm)   # ptr<3> → i32
    smem_base_i32 = llvm.PtrToIntOp(i32, smem_base_llvm).result   # ptr<3> → i32

    # Compute per-thread offset: base + tidx * 8  (8 bytes per u64)
    # tidx is a cutlass.Int32, grab its ir.Value
    tidx_val   = tidx # i32
    eight = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 8)).result

    byte_off = llvm.MulOp(tidx_val, eight, 0).result
    smem_addr  = llvm.AddOp(smem_base_i32, byte_off, 0).result  # per-thread addr

    # ---- 1. Read clock registers ----------------------------------------
    # mov.u32  %dest, %clock;   (low 32 bits)
    # mov.u32  %dest, %clock_hi; (high 32 bits)
    # Output constraint "=r" → 32-bit int register
    clock_lo = llvm.inline_asm(
        i32,
        [],
        asm_string="mov.u32 $0, %clock;",
        constraints="=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    clock_hi = llvm.inline_asm(
        i32,
        [],
        asm_string="mov.u32 $0, %clock_hi;",
        constraints="=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    cute.printf("cur {}: {} {}", tidx, clock_lo, clock_hi)

    # ---- 2. Combine clock_lo and clock_hi into b64 --------------------
    # b64 = (clock_hi << 32) | clock_lo
    clock_lo_i64 = llvm.ZExtOp(i64, clock_lo).result
    clock_hi_i64 = llvm.ZExtOp(i64, clock_hi).result
    thirty_two = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result
    clock_hi_shift = llvm.ShlOp(clock_hi_i64, thirty_two, 0).result
    clock_b64 = llvm.OrOp(clock_lo_i64, clock_hi_shift).result

    # ---- 3. REG → SMEM (st.shared.v2) ----------------------------------
    # For v2, we need to pass two 32-bit registers that form the 64-bit value
    llvm.inline_asm(
        None,
        [smem_addr, clock_lo, clock_hi],
        asm_string="st.shared.v2.b32 [$0], {$1, $2};",
        constraints="r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    # Optional: barrier so all threads have written before any reads back
    cute.arch.sync_threads()

    # ---- 4. SMEM → REG -> GMEM-------------------------------------------------
    loaded0 = llvm.inline_asm(
        i64,
        [smem_addr],
        asm_string="ld.shared.b32 $0, [$1];",
        constraints="=l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    # out is a cute.Tensor (generic ptr, addrspace 0 after DLPack conversion)
    # out.iterator is the cute.Pointer; its ir_value is !llvm.ptr (generic)
    gmem_base_llvm = out.iterator.llvm_ptr
    gmem_base_i64  = llvm.PtrToIntOp(i64, gmem_base_llvm).result

    # Per-thread byte offset in i64 (8 bytes for u64)
    tidx_i64  = llvm.ZExtOp(i64, tidx_val).result
    eight_i64  = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 8)).result
    byte_off64 = llvm.MulOp(tidx_i64, eight_i64, 0).result
    gmem_addr0  = llvm.AddOp(gmem_base_i64, byte_off64, 0).result

    llvm.inline_asm(
        None,
        [gmem_addr0, loaded0],
        asm_string="st.global.b32 [$0], $1;",
        constraints="l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    four = llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, 4)).result
    smem_addr1 = llvm.AddOp(smem_addr, four, 0).result  # per-thread addr
    loaded1 = llvm.inline_asm(
        i64,
        [smem_addr1],
        asm_string="ld.shared.b32 $0, [$1];",
        constraints="=l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    four_i64  = llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 4)).result
    gmem_addr1  = llvm.AddOp(gmem_addr0, four_i64, 0).result

    llvm.inline_asm(
        None,
        [gmem_addr1, loaded1],
        asm_string="st.global.b32 [$0], $1;",
        constraints="l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@cute.jit
def clock(out: cute.Tensor):

    print(f"[DSL INFO] output tensors:")
    print(f"[DSL INFO]   out = {out}")

    kernel = clock_kernel(out)
    grid = (1, 1, 1)
    block = (NUM_THREADS, 1, 1)
    print(f"grid: {grid}")
    print(f"block: {block}")
    kernel.launch(
        grid=grid,
        block=block,
    )

# ---------------------------------------------------------------------------
# 3.  Host launch wrapper
# ---------------------------------------------------------------------------
def run_clock_kernel(num_threads: int = NUM_THREADS, num_blocks: int = 1):
    """
    Allocates a uint32 output tensor on GPU, launches the kernel,
    returns the result as a CPU torch tensor.
    """
    assert num_threads == NUM_THREADS, (
        f"SharedStorage was compiled for NUM_THREADS={NUM_THREADS}; "
        f"pass num_threads={NUM_THREADS} or recompile."
    )

    # Allocate output: (num_blocks * num_threads,) u64 on GPU
    # torch doesn't have uint64 natively; use int64 (same bits)
    total = num_blocks * num_threads
    out_gpu = torch.zeros(total, dtype=torch.int64, device="cuda")

    # Convert to cute.Tensor via DLPack
    out_cute = from_dlpack(out_gpu, assumed_align=4)

    # Launch: grid=(num_blocks, 1, 1), block=(num_threads, 1, 1)
    clock_ = cute.compile(clock, out_cute)
    clock_(out_cute)

    torch.cuda.synchronize()

    # Reinterpret as uint64 for display (view as unsigned)
    out_cpu = out_gpu.cpu().view(torch.int64)   # same bits, signed view
    return out_cpu


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running clock_kernel: {1} block × {NUM_THREADS} threads")
    clocks = run_clock_kernel(num_threads=NUM_THREADS, num_blocks=1)

    print(f"\nClock values per thread (int64 view of u64 bits):")
    print(clocks)

    # Sanity: values should be non-zero and roughly increasing with tid
    # (monotonically only if threads execute serially; in practice they
    #  are clustered around the same cycle because warps run in lockstep)
    nonzero = (clocks != 0).sum().item()
    print(f"\nNon-zero entries: {nonzero} / {NUM_THREADS}")
    assert nonzero > 0, "All clock values are zero — something went wrong!"

    print("\nTest PASSED ✓")
