/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <cuda_runtime.h>

namespace torch { namespace mpi { namespace thc { namespace detail {

template<typename ScalarType>
void reduce(ScalarType* out, const ScalarType* in, unsigned long size, cudaStream_t stream);

}}}}
