--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
local MPI = require("torchmpi.env")

local cache = {}

-- Depending on the type of memory allocated on either the CPU or GPU, the
-- parameterserver operations clone and memoize tensors.
-- As a consequence, we provide an API to free the references to the copies we
-- maintain for collectives.
-- After calling those free functions, you should consider whether you want to
-- call collectgarbage twice.
cache.tensorReferences = {}
cache.parameterServers = {}
cache.sendTensorReferences = {}
cache.prefetchTensorReferences = {}
cache.freeReferencesToTensor = function(t)
   for k, v in pairs(cache.tensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.tensorReferences[k] = nil
      end
   end
   for k, v in pairs(cache.sendTensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.sendTensorReferences[k] = nil
      end
   end
   for k, v in pairs(cache.prefetchTensorReferences) do
      if k == t or v.orig == t or v.converted == t then
         cache.prefetchTensorReferences[k] = nil
      end
   end
   for k, ps in pairs(cache.parameterServers) do
      if k == t or v.orig == t or v.converted == t then
         cache.parameterserverfree(ps)
         cache.parameterServers[k] = nil
      end
   end
end

cache.freeAllTensorReferences = function()
   cache.tensorReferences = {}
   for t, ps in pairs(cache.parameterServers) do
      cache.parameterserverfree(ps)
   end
   cache.parameterServers = {}
   cache.sendTensorReferences = {}
   cache.prefetchTensorReferences = {}
end

require('ffi')
cache.freeDescriptors = function()
   MPI.C.torchmpi_free_ipc_descriptors()
end

return cache
