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

def cute_wgmma_sm90(
        m, n, k, transA, transB):
    return cutlass_tutorial.cute_wgmma_sm90(
            m, n, k,
            'T' if transA else 'N',
            'T' if transB else 'N'
            )

def cute_wgmma_tma_sm90(
        m, n, k, transA, transB):
    return cutlass_tutorial.cute_wgmma_tma_sm90(
            m, n, k,
            'T' if transA else 'N',
            'T' if transB else 'N'
            )

if __name__ == "__main__":
    #print(basic_gemm(1024, 1024, 1024, 1.0, 0.0))
    #print(cutlass_utilities(1024, 1024, 1024, 1.0, 0.0))
    #print(cute_wgmma_sm90(5120, 5120, 4096, True, False))
    print(cute_wgmma_tma_sm90(5120, 5120, 4096, True, False))
