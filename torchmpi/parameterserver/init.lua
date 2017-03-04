--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]

require('torch')
local MPI = require('torchmpi.env')
local wrap = require('torchmpi.wrap')
local cache = require('torchmpi.cache')

local PS = require('torchmpi.parameterserver.env')

require("torchmpi.parameterserver.update")
require("torchmpi.parameterserver.downpourupdate")
require("torchmpi.parameterserver.easgdupdate")

-- Creates a parameter server from a tensor, allocating space for and copying
-- the local portion (shard) of the tensor:data().
--
-- parameterserver_init is a blocking collective operation that must be run on
-- all processes in the communicator so they can synchronize and shard properly.
--
-- For each such process, a local shard is inited with the
-- *corresponding values of of that process*.
-- In particular, suppose process_i has a local tensor_i uniformly initialized
-- to value 'i'.
-- Then, after running PS.init(tensor_i) on each process_i,
-- the distributed parameter server will contain
--    shard[0]   shard[1]    ...  shard[MPI.size() - 1]
--       0          1                MPI.size() - 1
-- If you want a uniform sharded tensor initialization, you can make all processes
-- synchronize to some value first with a broadcast then init the parameterserver.
--
-- Details:
--   1. A parameterserver allocates and owns a lcoal shard for holding the parameters as well
--      as a local buffer shard for send/receives.
--      This local buffer shard is tied to the shape of the tensor, its size is determined
--      by sharding the tensor on all the processes participating in the MPI communicator.
--   2. Parameter server initialization is a blocking operation (synchronized with MPI.barriers)
--      Therefore all ranks must initialize all parameterservers for matching tensors in the
--      same order. The ordering property is important because MPI tags have to match.
--   3. In geenral, you can use a parameterserver inited with a tensor t1 to synchronize
--      with a tensor t2:
--      - if t1 and t2 have the same number of elements (see below).
--      - if t1 and t2 are contiguous tensors
PS.init = function(t)
   assert(t:nElement() >= MPI.size(),
          'NYI: corner case where nElement < MPI.size()')
   local fun = 'torchmpi_parameterserver_init_TH'..torch.type(t):gsub('torch.', '')
   return wrap.executeTensorFun(fun, t)
end

PS.send = function(ps, t, rule)
   local rule = rule or 'none'
   assert(type(rule) == 'string',
          'Usage parameterserver.send(cdata<void*>, *Tensor, string)' ..
             ' but parameter #2 is: ' .. type(rule))
   assert(type(ps) == 'cdata',
          'Usage parameterserver.send(cdata<void*>, *Tensor, string)' ..
             ' but parameter #2 is: ' .. type(ps))
   local fun = 'torchmpi_parameterserver_send_TH'..torch.type(t):gsub('torch.', '')
   return wrap.executePSFun(fun, ps, t, rule or 'none')
end

PS.receive = function(ps, t)
   assert(type(ps) == 'cdata',
          'Usage parameterserver.send(cdata<void*>, *Tensor)' ..
             ' but parameter #2 is: ' .. type(ps))
   local fun = 'torchmpi_parameterserver_receive_TH'..torch.type(t):gsub('torch.', '')
   return wrap.executePSFun(fun, ps, t)
end

PS.free = function(ps)
   assert(type(ps) == 'cdata',
          'Usage parameterserver.send(cdata<void*>, *Tensor)' ..
             ' but parameter #2 is: ' .. type(ps))
   MPI.C.torchmpi_parameterserver_free(ps)
end

PS.freeAll = function()
   MPI.C.torchmpi_parameterserver_free_all()
end

PS.syncHandle = function(sh)
   return MPI.C.torchmpi_parameterserver_synchronize_handle(sh);
end


-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- The following is a set of helper functions that operate on lists of tensor
-- and save bookkeeping information in torchmpi.cache.
-- They make implementation choices which may not work for you but at least
-- they allow to build a simple dataparallel, downpour and easgd
-- (see examples/mnist/mnist_parameterserver_*).
--
-- Be sure to understand the assumptions if you use them.
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- Takes a table of tensors, usually coming from net:parameters()
-- Can be either weights or gradients
local function sanityCheckTensors(tensors)
   for i, t in ipairs(tensors) do
      assert(torch.isTensor(t), 'Not a tensor ' .. torch.type(t))
      assert(cache.prefetchTensorReferences[t],
             'No prefetchTensor for tensor @' .. tostring(t:cdata()))
      assert(cache.parameterServers[t],
             'No parameterserver for tensor @' .. tostring(t:cdata()))
   end
end

-- Initialize on parameterserver client/server pair for each tensor in the
-- list
-- Takes:
-- 1. tensors: a table of tensors, usually coming from net:parameters().
--    Can be either weights or gradients
-- 2. prefetchAllocator: a prefetch alloc function to control where the prefetch
--    tensor should live, if for instance you are short for GPU memory.
--    Defaults to a simple tensor:clone function.
-- 3. psInitFun: a parameter server init function which informs how to
--    intialize the sharded tensor. Defaults to copying the tensor from the
--    rank 0.
PS.initTensors = function(tensors, prefetchAllocator, psInitFun)
   local prefetchAllocator = prefetchAllocator or
      function(t)
         local cputensor = torch.type(t):find('Cuda') and
            cutorch.createCudaHostTensor():resize(t:nElement()):copy(t) or
            t:clone()
         return cputensor
      end

   local psInitFun = function(ps, t)
      if MPI.rank() == 0 then
         PS.syncHandle(PS.send(ps, t, 'copy'))
      end
      MPI.barrier()
   end

   for i, t in ipairs(tensors) do
      assert(not cache.parameterServers[t])
      cache.prefetchTensorReferences[t] = prefetchAllocator(t)
      cache.parameterServers[t] = PS.init(cache.prefetchTensorReferences[t])
      psInitFun(cache.parameterServers[t], cache.prefetchTensorReferences[t])
   end
   sanityCheckTensors(tensors)
end

-- Takes a table of tensors, usually coming from net:parameters()
-- For each tensor t, receives the corresponding shards from
-- cache.parameterServers[t] into a local prefetched tensor
-- cache.prefetchTensorReferences[t].
-- Returns a table of handles on which you can run PS.syncHandle to make sure
-- the prefetch is complete.
PS.prefetchTensors = function(tensors)
   sanityCheckTensors(tensors)
   local handles = {}
   for i, t in ipairs(tensors) do
      table.insert(handles,
                   PS.receive(
                      cache.parameterServers[t],
                      cache.prefetchTensorReferences[t]))
   end
   return handles
end

-- Takes a table of tensors, usually coming from net:parameters()
-- Takes a localUpdateRule function(prefetched, t) which applies prefetched
-- to t.
PS.integrateTensors = function(tensors, localUpdateRule)
   sanityCheckTensors(tensors)
   for i, t in ipairs(tensors) do
      localUpdateRule(cache.prefetchTensorReferences[t], t)
   end
end

-- Takes a table of tensors, usually coming from net:parameters()
-- Can be either weights or gradients
-- Optionally takes a localUpdateRule function(prefetched, t) which
-- applies the prefetched to t.
-- Returns a table of handles on which you can run PS.syncHandle to make
-- sure the send is complete.
PS.sendTensors = function(tensors, updates, remoteUpdateRule, localPreprocessFun)

   local sendAllocator =
      function(t)
         local cputensor = torch.type(t):find('Cuda') and
            cutorch.createCudaHostTensor():resize(t:nElement()):copy(t) or
            t
         return cputensor
      end

   sanityCheckTensors(tensors)
   localPreprocessFun = localPreprocessFun or function(t) return t end

   local sync = false
   local handles = {}
   for i, t in ipairs(updates) do
      cache.sendTensorReferences[t] = cache.sendTensorReferences[t] or
         sendAllocator(t)
      if cache.sendTensorReferences[t] ~= t then
         -- Only taken for cuda tensors that have been copied to pinned
         cache.sendTensorReferences[t]:copyAsync(t)
         sync = true
      end
   end
   if sync then cutorch.synchronize() end
   for i, t in ipairs(updates) do
      table.insert(handles,
                   PS.send(cache.parameterServers[tensors[i]],
                           localPreprocessFun(cache.sendTensorReferences[t]),
                           remoteUpdateRule))
   end
   return handles
end

PS.syncHandles = function(handles)
   for i, h in ipairs(handles) do
      PS.syncHandle(h)
   end
   return {}
end

return PS
