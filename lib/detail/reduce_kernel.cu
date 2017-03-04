#include "THC.h"

#include "reduce_kernel.h"
#include "device_functions.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cassert>
#include <iostream>
#include <cstdio>

using namespace std;

namespace torch { namespace mpi { namespace thc { namespace detail {

namespace {

inline __host__ __device__ void operator+=(float4 &a, float4& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

template<typename ScalarType>
void __global__ reduceKernelBaseline(
    ScalarType* out, const ScalarType*__restrict__ in, unsigned long size) {
  int scalarStart = blockIdx.x * blockDim.x + threadIdx.x;
  int scalarInc = blockDim.x * gridDim.x;
  for (int i = scalarStart; i < size; i += scalarInc) {
    out[i] += in[i];
  }
}

// TODO: unroll by 4!
template<typename ScalarType, typename VectorType, int NScalarInVector>
void __global__ reduceKernel(
    ScalarType* out,
    const ScalarType*__restrict__ in,
    unsigned long prologue,
    unsigned long kernel,
    unsigned long epilogue) {
  const ScalarType*__restrict__ inEnd = in + prologue + kernel + epilogue;
  ScalarType* outEnd = out + prologue + kernel + epilogue;

  int scalarStart = blockIdx.x * blockDim.x + threadIdx.x;
  int scalarInc = blockDim.x * gridDim.x;
  int vectorStart = scalarStart * NScalarInVector;
  int vectorInc = scalarInc * NScalarInVector;
  // Prologue
  for (int i = scalarStart; i < prologue; i += scalarInc) {
    out[i] += in[i];
  }

  // Kernel
  in += prologue + vectorStart;
  out += prologue + vectorStart;
  for ( ; in + NScalarInVector <= inEnd; ) {
    const VectorType* iip = reinterpret_cast<const VectorType*>(in);
    VectorType* oop = reinterpret_cast<VectorType*>(out);
#if __CUDA_ARCH__ >= 350
    VectorType ii = __ldg(const_cast<VectorType*>(iip));
#else
    VectorType ii = *iip;
#endif
    *oop += ii;
    out += vectorInc;
    in += vectorInc;
  }

  // Epilogue
  in = inEnd - epilogue + scalarStart;
  out = outEnd - epilogue + scalarStart;
  for ( ; in < inEnd; out += scalarInc, in += scalarInc) {
    *out += *in;
  }
}

template<typename ScalarType, typename VectorType, int NScalarInVector>
void reduceSpec(
    ScalarType* out,
    const ScalarType* in,
    unsigned long size,
    cudaStream_t stream)
{
  constexpr int VectorSize = sizeof(VectorType);
  THAssert(
    reinterpret_cast<uintptr_t>(in) % VectorSize ==
    reinterpret_cast<uintptr_t>(out) % VectorSize);

  unsigned long align, prologue, kernel, epilogue;
  align = reinterpret_cast<uintptr_t>(out) % VectorSize;
  THAssert(align < VectorSize);
  prologue = align ? (VectorSize - align) : 0;
  THAssert(prologue % sizeof(ScalarType) == 0);
  prologue = prologue / sizeof(ScalarType);

  if (prologue < size) {
    align = reinterpret_cast<uintptr_t>(out + size) % VectorSize;
    epilogue = align / sizeof(ScalarType);
    THAssert(align % sizeof(ScalarType) == 0);
    kernel = size - prologue - epilogue;
    THAssert((kernel * sizeof(ScalarType)) % VectorSize == 0);
  } else {
    prologue = size;
    kernel = 0;
    epilogue = 0;
  }
  THAssert(prologue + kernel + epilogue == size);

  // 2 SMs seems enough to saturate the BW at the larger sizes we are
  // interested in.
  dim3 threads(1024);
  reduceKernel<ScalarType, VectorType, NScalarInVector>
    <<<dim3(2), threads, 0, stream>>>
    (out, in, prologue, kernel, epilogue);
}

}

template<typename ScalarType>
void reduce(ScalarType* out, const ScalarType* in, unsigned long size, cudaStream_t stream) {
  dim3 threads(1024);
  reduceKernelBaseline<ScalarType>
    <<<dim3(1), threads, 0, stream>>> (out, in, size);
}

template void reduce<unsigned char>(unsigned char* out, const unsigned char* in, unsigned long size, cudaStream_t stream);
template void reduce<char>(char* out, const char* in, unsigned long size, cudaStream_t stream);
template void reduce<short>(short* out, const short* in, unsigned long size, cudaStream_t stream);
template void reduce<int>(int* out, const int* in, unsigned long size, cudaStream_t stream);
template void reduce<long>(long* out, const long* in, unsigned long size, cudaStream_t stream);
//template void reduce<float>(float* out, const float* in, unsigned long size, cudaStream_t stream);
template void reduce<double>(double* out, const double* in, unsigned long size, cudaStream_t stream);

template<> void reduce(float* out, const float* in, unsigned long size, cudaStream_t stream) {
  reduceSpec<float, float4, 4>(out, in, size, stream);
}

}}}}
