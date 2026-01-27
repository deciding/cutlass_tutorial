import time
import torch
import cutlass.cute as cute
#from sol0 import solve
#from sol1 import solve
from sol2 import solve


# -------------------------------
# Reference implementation
# -------------------------------
def reference_vector_add(A, B):
    return A + B


# -------------------------------
# Verifier runner
# -------------------------------
def verify_vector_add(
    N,
    dtype=torch.float32,
    warmup=10,
    iters=100,
    device="cuda"
):
    results = []

    print(f"\nVerifying N = {N}")

    # Allocate PyTorch tensors
    A_torch = torch.rand(N, device=device, dtype=dtype)
    B_torch = torch.rand(N, device=device, dtype=dtype)
    C_torch = torch.empty(N, device=device, dtype=dtype)

    # Wrap as CuTe tensors
    A = cute.runtime.from_dlpack(A_torch)
    B = cute.runtime.from_dlpack(B_torch)
    C = cute.runtime.from_dlpack(C_torch)

    N_cute = cute.Uint32(N)

    # -------------------------------
    # Correctness check
    # -------------------------------
    solve(A, B, C, N_cute)

    torch.cuda.synchronize()

    C_ref = reference_vector_add(A_torch, B_torch)

    max_err = (C_torch - C_ref).abs().max().item()
    print(f"Max error: {max_err:.3e}")

    if max_err > 1e-6:
        raise RuntimeError("Verification failed")

# -------------------------------
# Benchmark runner
# -------------------------------
def benchmark_vector_add(
    shapes,
    dtype=torch.float32,
    warmup=10,
    iters=100,
    device="cuda"
):
    results = []

    for N in shapes:
        print(f"\nBenchmarking N = {N}")

        # Allocate PyTorch tensors
        A_torch = torch.rand(N, device=device, dtype=dtype)
        B_torch = torch.rand(N, device=device, dtype=dtype)
        C_torch = torch.empty(N, device=device, dtype=dtype)

        # Wrap as CuTe tensors
        A = cute.runtime.from_dlpack(A_torch)
        B = cute.runtime.from_dlpack(B_torch)
        C = cute.runtime.from_dlpack(C_torch)

        N_cute = cute.Uint32(N)

        # -------------------------------
        # Warm-up
        # -------------------------------
        for _ in range(warmup):
            solve(A, B, C, N_cute)

        torch.cuda.synchronize()

        # -------------------------------
        # Timing
        # -------------------------------
        start = time.perf_counter()
        for _ in range(iters):
            solve(A, B, C, N_cute)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / iters  # seconds

        # -------------------------------
        # Performance metrics
        # -------------------------------
        bytes_moved = N * 3 * A_torch.element_size()
        bandwidth_gbps = bytes_moved / avg_time / 1e9

        print(f"Avg time: {avg_time * 1e6:.2f} us")
        print(f"Effective bandwidth: {bandwidth_gbps:.2f} GB/s")

        results.append({
            "N": N,
            "time_us": avg_time * 1e6,
            "bandwidth_GBps": bandwidth_gbps,
        })

    return results


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    shapes = [
        1 << 10,    # 1K
        1 << 15,    # 32K
        1 << 20,    # 1M
        1 << 24,    # 16M
    ]

    verify_vector_add(2048)
    exit()
    results = benchmark_vector_add(shapes)

    print("\nSummary:")
    for r in results:
        print(r)

