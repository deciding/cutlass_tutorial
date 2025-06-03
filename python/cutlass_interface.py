from typing import Optional, Tuple

import torch

import cutlass_tutorial


def basic_gemm(
    M, N, K, alpha, beta,
):
    return cutlass_tutorial.basic_gemm(M, N, K, alpha, beta)

def cutlass_utilities(
    M, N, K, alpha, beta,
):
    return cutlass_tutorial.cutlass_utilities(M, N, K, alpha, beta)

if __name__ == "__main__":
    print(basic_gemm(1024, 1024, 1024, 1.0, 0.0))
    print(cutlass_utilities(1024, 1024, 1024, 1.0, 0.0))
