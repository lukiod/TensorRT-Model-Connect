/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Links the real flux gpu_matmul translation unit against CPU CUDA and cuBLAS
// stubs, so an allocation failure can be injected without a GPU. Covers the
// case where a failed grow previously committed the requested size anyway,
// leaving the workspace claiming capacity it did not have.

#include "families/flux/runtime/gpu_matmul.h"

#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

bool g_fail_next_allocation = false;
std::set<void*> g_outstanding;
std::uintptr_t g_next_address = 0x1000;
int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

} // namespace

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    (void)size;
    if (g_fail_next_allocation) {
        *devPtr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    void* address = reinterpret_cast<void*>(g_next_address);
    g_next_address += 0x1000;
    g_outstanding.insert(address);
    *devPtr = address;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    if (devPtr != nullptr) {
        g_outstanding.erase(devPtr);
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) {
    return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return cudaSuccess;
}
cudaError_t cudaStreamCreate(cudaStream_t*) {
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t) {
    return cudaSuccess;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    *handle = reinterpret_cast<cublasHandle_t>(0x1);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDestroy_v2(cublasHandle_t) {
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetStream_v2(cublasHandle_t, cudaStream_t) {
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                              const float*, const float*, int, const float*, int, const float*,
                              float*, int) {
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"

namespace {

// Drives the public entry point, which grows the three workspace buffers.
void run_matmul(int32_t m, int32_t k, int32_t n) {
    const std::vector<float> a(static_cast<std::size_t>(m) * k, 1.0F);
    const std::vector<float> b(static_cast<std::size_t>(k) * n, 1.0F);
    std::vector<float> out(static_cast<std::size_t>(m) * n, 0.0F);
    trtmc::flux_gpu_matmul_bias(a.data(), b.data(), nullptr, out.data(), m, k, n);
}

} // namespace

int main() {
    trtmc::flux_gpu_matmul_init();

    // A failed allocation must be reported, not silently recorded as capacity.
    g_fail_next_allocation = true;
    bool threw = false;
    try {
        run_matmul(4, 4, 4);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "a failed workspace allocation should throw");
    check(g_outstanding.empty(), "a failed first allocation should leave nothing allocated");

    // The failure must not have left the workspace claiming a size it never got:
    // the retry has to allocate for real rather than reuse a null pointer.
    g_fail_next_allocation = false;
    bool retry_threw = false;
    try {
        run_matmul(4, 4, 4);
    } catch (const std::runtime_error&) {
        retry_threw = true;
    }
    check(!retry_threw, "a retry after a failed allocation should succeed");
    check(!g_outstanding.empty(), "the retry should have allocated real buffers");
    const std::set<void*> after_success = g_outstanding;

    // A failed grow must keep the buffers the workspace already had.
    g_fail_next_allocation = true;
    bool grow_threw = false;
    try {
        run_matmul(64, 64, 64);
    } catch (const std::runtime_error&) {
        grow_threw = true;
    }
    check(grow_threw, "a failed grow should throw");
    check(g_outstanding == after_success, "a failed grow must not free the existing buffers");

    // And the workspace is still usable at its original size.
    g_fail_next_allocation = false;
    bool reuse_threw = false;
    try {
        run_matmul(4, 4, 4);
    } catch (const std::runtime_error&) {
        reuse_threw = true;
    }
    check(!reuse_threw, "the workspace should still be usable after a failed grow");

    trtmc::flux_gpu_matmul_shutdown();
    check(g_outstanding.empty(), "shutdown should release every buffer");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flux gpu matmul allocation-failure checks passed\n");
    return 0;
}
