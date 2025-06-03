#include <pybind11/pybind11.h>
#include "cutlass_all_functions.h"

int basic_gemm(int M, int N, int K, float alpha, float beta){
	return basic_gemm_internal(M, N, K, alpha, beta);
}

int cutlass_utilities(int M, int N, int K, float alpha, float beta){
	return cutlass_utilities_internal(M, N, K, alpha, beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Cutlass Tutorial";
    m.def("basic_gemm", &basic_gemm); // 00
    m.def("cutlass_utilities", &cutlass_utilities); // 01
}
