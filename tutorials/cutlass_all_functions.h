int basic_gemm_internal(int M, int N, int K, float alpha, float beta);
int cutlass_utilities_internal(int M, int N, int K, float alpha, float beta);

// cute
int cute_wgmma_sm90_internal(int m, int n, int k, char transA, char transB);
int cute_wgmma_tma_sm90_internal(int m, int n, int k, char transA, char transB);
