import time
import torch
import cutlass.cute as cute
#from sol1 import solve
from sol2 import solve


# -------------------------------
# Reference implementation
# -------------------------------
def reference_matmul(A, B):
    return A @ B


# -------------------------------
# Verifier runner
# -------------------------------
def verify_matmul(
    M,
    N,
    K,
    dtype=torch.float32,
    device="cuda",
):
    print(f"\nVerifying M={M}, N={N}, K={K}")

    # Allocate PyTorch tensors
    A_torch = torch.rand((M, K), device=device, dtype=dtype)
    B_torch = torch.rand((K, N), device=device, dtype=dtype)
    C_torch = torch.empty((M, N), device=device, dtype=dtype)

    # Wrap as CuTe tensors
    A = cute.runtime.from_dlpack(A_torch)
    B = cute.runtime.from_dlpack(B_torch)
    C = cute.runtime.from_dlpack(C_torch)

    M_cute = cute.Int32(M)
    N_cute = cute.Int32(N)
    K_cute = cute.Int32(K)

    # -------------------------------
    # Correctness check
    # -------------------------------
    solve(A, B, C, M_cute, N_cute, K_cute)
    torch.cuda.synchronize()

    C_ref = reference_matmul(A_torch, B_torch)

    max_err = (C_torch - C_ref).abs().max().item()
    print(f"Max error: {max_err:.3e}")

    if max_err > 1e-5:
        raise RuntimeError("Verification failed")


# -------------------------------
# Benchmark runner
# -------------------------------
def benchmark_matmul(
    shapes,
    dtype=torch.float32,
    warmup=10,
    iters=100,
    device="cuda",
):
    results = []

    for (M, N, K) in shapes:
        print(f"\nBenchmarking M={M}, N={N}, K={K}")

        # Allocate PyTorch tensors
        A_torch = torch.rand((M, K), device=device, dtype=dtype)
        B_torch = torch.rand((K, N), device=device, dtype=dtype)
        C_torch = torch.empty((M, N), device=device, dtype=dtype)

        # Wrap as CuTe tensors
        A = cute.runtime.from_dlpack(A_torch)
        B = cute.runtime.from_dlpack(B_torch)
        C = cute.runtime.from_dlpack(C_torch)

        M_cute = cute.Int32(M)
        N_cute = cute.Int32(N)
        K_cute = cute.Int32(K)

        # -------------------------------
        # Warm-up
        # -------------------------------
        for _ in range(warmup):
            solve(A, B, C, M_cute, N_cute, K_cute)
        torch.cuda.synchronize()

        # -------------------------------
        # Timing
        # -------------------------------
        start = time.perf_counter()
        for _ in range(iters):
            solve(A, B, C, M_cute, N_cute, K_cute)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / iters  # seconds

        # -------------------------------
        # Performance metrics
        # -------------------------------
        flops = 2 * M * N * K
        tflops = flops / avg_time / 1e12

        bytes_moved = (
            M * K * A_torch.element_size()
            + K * N * B_torch.element_size()
            + M * N * C_torch.element_size()
        )
        bandwidth_gbps = bytes_moved / avg_time / 1e9

        print(f"Avg time: {avg_time * 1e6:.2f} us")
        print(f"TFLOP/s: {tflops:.2f}")
        print(f"Effective bandwidth: {bandwidth_gbps:.2f} GB/s")

        results.append({
            "M": M,
            "N": N,
            "K": K,
            "time_us": avg_time * 1e6,
            "tflops": tflops,
            "bandwidth_GBps": bandwidth_gbps,
        })

    return results


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Small correctness test
    verify_matmul(M=128, N=128, K=128)
    #exit()

    # Benchmark shapes
    shapes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    results = benchmark_matmul(shapes)

    print("\nSummary:")
    for r in results:
        print(r)

