import torch
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

##############################################
# Naive add
##############################################

@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    # Map logical index to physical address via tensor layout
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    # Perform element-wise addition
    gC[mi, ni] = a_val + b_val

@cute.jit
def naive_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    num_threads_per_block = 256

    print(f"[DSL INFO] Input tensors:")
    print(f"[DSL INFO]   mA = {mA}")
    print(f"[DSL INFO]   mB = {mB}")
    print(f"[DSL INFO]   mC = {mC}")

    m, n = mA.shape
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    grid = ((m * n) // num_threads_per_block, 1, 1)
    block = (num_threads_per_block, 1, 1)
    print(f"grid: {grid}")
    print(f"block: {block}")
    kernel.launch(
        grid=grid,
        block=block,
    )

##############################################
# Vectorized add
##############################################

@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape[1]  # thread-domain
    ni = thread_idx % n
    mi = thread_idx // n

    # Map logical index to physical address via tensor layout
    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")
    print(f"[DSL INFO] sliced gB = {gB[(None, (mi, ni))]}")

    # Perform element-wise addition
    gC[(None, (mi, ni))] = a_val + b_val

@cute.jit
def vectorized_elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    threads_per_block = 256

    gA = cute.zipped_divide(mA, (1, 4))
    gB = cute.zipped_divide(mB, (1, 4))
    gC = cute.zipped_divide(mC, (1, 4))

    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    dim1_size = cute.size(gC, mode=[1])
    grid = ( dim1_size// threads_per_block, 1, 1)
    block = (threads_per_block, 1, 1)
    print(f"grid: {grid}")
    print(f"block: {block}")
    vectorized_elementwise_add_kernel(gA, gB, gC).launch(
        grid=grid,
        block=block,
    )

##############################################
# TV add
##############################################
@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # --------------------------------
    # slice for thread-block level view
    # --------------------------------
    blk_coord = ((None, None), bidx)

    # For each Threablock
    # logical coord -> address
    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address
    blkB = gB[blk_coord]  # (TileM, TileN) -> physical address
    blkC = gC[blk_coord]  # (TileM, TileN) -> physical address

    # --------------------------------
    # compose for thread-index & value-index to physical mapping
    # --------------------------------
    # blockA:    (TileM, TileN) -> physical address
    # tv_layout: (tid, vid)     -> (TileM, TileN)
    # tidfrgA = blkA o tv_layout
    # tidfrgA:   (tid, vid) -> physical address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print(f"Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}") # ((32,4),(8,4)):((8,8192),(1,2048))

    # --------------------------------
    # slice for thread-level view
    # --------------------------------
    # `None` represent slice of the entire per-thread data
    thr_coord = (tidx, None)

    # slice for threads: vid -> address
    thrA = tidfrgA[thr_coord]  # (V) -> physical address
    thrB = tidfrgB[thr_coord]  # (V) -> physical address
    thrC = tidfrgC[thr_coord]  # (V) -> physical address

    print(f"thrA/B/C:")
    print(f"  thrA: {thrA.type}") # ((8,4)):((1,2048)), 32 values per thread to physical addr
    print(f"  thrB: {thrB.type}")
    print(f"  thrC: {thrC.type}")

    #thrC[None] = thrA.load() + thrB.load()
    tidfrgC[thr_coord] = thrA.load() + thrB.load()

@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA layout: (M, N):(N, 1)
    # TV layout map thread & value index to (16, 256) logical tile
    #  - contiguous thread index maps to mode-1 because input layout is contiguous on
    #     mode-1 for coalesced load-store
    #  - each thread load 8 contiguous element each row and load 4 rows
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}") # 16, 256
    # tv -> mn, with mn as col-major
    print(f"TV Layout: {tv_layout}") # ((32,4),(8,4)):((128,4),(16,1))

    print(f"Global mem")
    print(f"  mA: {mA.type}") # 2048, 2048 : 2048, 1
    print(f"  mB: {mB.type}")
    print(f"  mC: {mC.type}")
    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"Tiled Input Tensors:")
    print(f"  gA: {gA.type}") # ((16,256),(128,8)):((2048,1),(32768,256)), default m-major
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")

    num_blocks = cute.size(gC, mode=[1])
    num_threads = cute.size(tv_layout, mode=[0])
    print(f"num_blocks: {num_blocks}")
    print(f"num_threads: {num_threads}")
    # Launch the kernel asynchronously
    # Async token(s) can also be specified as dependencies
    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
    )


if __name__ == "__main__":
    M, N = 2048, 2048

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    # Compile kernel
    #naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)
    #naive_elementwise_add_(a_, b_, c_)

    #vectorized_elementwise_add_ = cute.compile(vectorized_elementwise_add, a_, b_, c_)
    #vectorized_elementwise_add_(a_, b_, c_)

    elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
    elementwise_add_(a_, b_, c_)

    # verify correctness
    torch.testing.assert_close(c, a + b)
