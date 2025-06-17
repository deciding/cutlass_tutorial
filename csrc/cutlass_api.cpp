#include <pybind11/pybind11.h>
#include "cutlass_all_functions.h"

int basic_gemm(int M, int N, int K, float alpha, float beta){
	return basic_gemm_internal(M, N, K, alpha, beta);
}

int cutlass_utilities(int M, int N, int K, float alpha, float beta){
	return cutlass_utilities_internal(M, N, K, alpha, beta);
}

int cute_wgmma_sm90(int m, int n, int k, char transA, char transB){
    return cute_wgmma_sm90_internal(m, n, k, transA, transB);
}

int cute_wgmma_tma_sm90(int m, int n, int k, char transA, char transB){
    return cute_wgmma_tma_sm90_internal(m, n, k, transA, transB);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Cutlass Tutorial";
    m.def("basic_gemm", &basic_gemm); // 00
    m.def("cutlass_utilities", &cutlass_utilities); // 01
    m.def("cute_wgmma_sm90", &cute_wgmma_sm90);
    m.def("cute_wgmma_tma_sm90", &cute_wgmma_tma_sm90);
}
