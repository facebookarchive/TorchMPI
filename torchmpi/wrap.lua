--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
local MPI = require("torchmpi.env")

local M = {}

-- Helper functions which hides some implementation details
M.executePSFun = function(funname, ps, t, ...)
   if torch.type(t):find('Cuda') then
      return MPI.C[funname](ps, cutorch.getState(), t:cdata(), ...)
   end
   return MPI.C[funname](ps, t:cdata(), ...)
end

M.executeTensorFun = function(funname, t, ...)
   if torch.type(t):find('Cuda') then
      return MPI.C[funname](cutorch.getState(), t:cdata(), ...)
   end
   return MPI.C[funname](t:cdata(), ...)
end

M.executeMPICFun = function(funname, t, ...)
   if torch.type(t):find('Cuda') then
      return MPI.C[funname](cutorch.getState(), t:cdata(), ...)
   end
   return MPI.C[funname](t:cdata(), ...)
end

M.executeMPICFun2 = function(funname, t1, t2, ...)
   return M.executeMPICFun(funname, t1, t2:cdata(), ...)
end

-- TODO: Fixme: there is a deadlock when trying to decompose
-- a GPU transfer into smaller asynchronous ones.
-- Works fine for CPU
M.executeMPIChunkedAllreduceCFun = function(funname, t1, t2)
   if true then
      return M.executeMPICFun2(funname, t1, t2)
   end

   if t1:nElement() <= MPI.C.torchmpi_get_small_gpu_allreduce_size() then
      -- Latency-bound, don't split up, just run in sync
      funname = funname:gsub('_async', '')
      return M.executeMPICFun2(funname, t1, t2)
   end
   local handles = {}
   local nt = 2 * MPI.C.torchmpi_get_collective_num_threads()
   local chunk = math.max(math.ceil(t1:nElement() / nt), 2^20)
   local size = t1:nElement()
   local step = 0
   for i = 1, size, chunk do
      local len = math.min(size - i + 1, chunk)
      local tt1 = t1:narrow(1, i, len)
      local tt2 = (t1 == t2) and tt1 or t2:narrow(1, i, len)
      local h = M.executeMPICFun2(funname, tt1, tt2)
      table.insert(handles, h)
      step = step + 1
   end
   assert(step <= nt) -- sanity check
   -- TODO: once async works properly combine handles
   for i, h in ipairs(handles) do
      handles[i] = MPI.syncHandle(handles[i])
   end
   return handles[1]
end

return M
