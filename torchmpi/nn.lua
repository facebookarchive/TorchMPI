--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')
local mpi = require('torchmpi.env')
local mpicache = require('torchmpi.cache')
require('torchmpi.BlockSequential')

local __debug__ = false

local M = {}

local selectCollective = function(tensor, sync, collective)
   local ttype = torch.type(tensor):find('Cuda') and 'gpu' or 'cpu'
   if mpi.needInterNodeCollectives then
      return assert(mpi.collectiveSelector[ttype].multinode[sync][collective],
      'Could not find collective ' .. ttype .. ' multinode ' .. sync .. ' ' .. collective)
   else
      return assert(mpi.collectiveSelector[ttype].singlenode[sync][collective],
      'Could not find collective ' .. ttype .. ' singlenode ' .. sync .. ' ' .. collective)
   end
end

-------------------------------- Synchronous operations ------------------------

-- Synchronize (default is Bcast from 0)
M.synchronizeParameters = function(net, allreduce)
   if not net.parameters then return end
   local allreduce = allreduce or false
   local p, g = net:parameters()
   for i, w in ipairs(p) do
      if allreduce then
         local allreduceTensor = selectCollective(w, 'sync', 'allreduceTensor')
         allreduceTensor(w)
         w:div(mpi.size())
      else
         local broadcastTensor = selectCollective(w, 'sync', 'broadcastTensor')
         broadcastTensor(0, w)
      end
   end
end

-- Synchronize gradients
M.synchronizeGradients = function(net)
   if not net.parameters then return end
   local p, g = net:parameters()
   for i, gw in ipairs(g) do
      local allreduceTensor = selectCollective(gw, 'sync', 'allreduceTensor')
      allreduceTensor(gw)
   end
end

-- Sanity checks
M.checkWithAllreduce = function(net)
   net:apply(function(m)
      if torch.isTypeOf(m, 'nn.Container') then
         return
      end
      if not m.parameters then return end
      local p, g = m:parameters()
      if not g or #g == 0 then
         return
      end
      for i, w in ipairs(p) do
         mpi.checkWithAllreduce(w, tostring(m))
      end
   end)
end

------------------------------- Asynchronous operations ------------------------
mpicache.asyncDetails = mpicache.asyncDetails or {}
mpicache.asyncDetails.handles = mpicache.asyncDetails.handles or {}
mpicache.asyncDetails.counter = mpicache.asyncDetails.counter or 0
mpicache.asyncDetails.originalBackwards = mpicache.asyncDetails.originalBackwards or {}

local backwardSynchronizationImpl = nil

M.async = {}
-- List of modules for which we want to skip the collective
M.async.skipCollective = {}

M.async.synchronizeGradients = function()
   error('Asynchronous gradient synchronization is achieved by calling '..
         'registerAsyncMPIBackward(net) and only then synchronizeGradients')
end


M.async.unregisterAsyncMPIBackward = function(net)
   -- Dispatch to the proper backward registering function
   if torch.isTypeOf(net, 'nn.BlockSequential') then
      error('NYI: unregisterAsyncMPIBackward for nn.BlockSequential')
      registerBlockSequentialAsyncMPIBackward(net)
   else
      if torch.isTypeOf(net, 'nn.Container') then
         for _, m in ipairs(net:listModules()) do
            -- Freaking listModules also lists 'self', filter it out!
            if m ~= net then
               m.backward = mpicache.asyncDetails.originalBackwards[m] or m.backward
            end
         end
      elseif net.parameters then
         net.backward = mpicache.asyncDetails.originalBackwards[m] or net.backward
      end
   end
end

M.async.registerAsyncMPIBackward = function(net, syncGradientFrequency)
   local syncGradientFrequency = syncGradientFrequency or 1

   if mpi.withCuda and not mpi.async.streams then
      error("torchmpi.nn async backwards requires explicit registration of torchmpi.async.streams.\n"..
               "Please call torchmpi.async.initStreams(true) to initialize streams with cudaStreamsNonBlocking.\n"..
               "The reserved streams are:\n"..
               "\ttorchmpi.async.streams.defaultStream = 0\n"..
               "\ttorchmpi.async.streams.copyToGPUStream = 1\n"..
               "\ttorchmpi.async.streams.copyToCPUStream = 2\n"..
               "\ttorchmpi.async.streams.localCollectiveStream = 3\n"..
               "\ttorchmpi.async.streams.globalCollectiveStream = 4\n")
   end


   -- Regular modules async backward
   local function registerAsyncMPIBackward(m, incrementCounter)
      if torch.isTypeOf(m, 'nn.Container') then
         return
      end

      mpicache.asyncDetails.originalBackwards[m] =
         mpicache.asyncDetails.originalBackwards[m] or m.backward
      local function asyncMPIBackward(...)
         local res = mpicache.asyncDetails.originalBackwards[m](...)
         if incrementCounter then
            mpicache.asyncDetails.counter = mpicache.asyncDetails.counter + 1
         end
         if mpicache.asyncDetails.counter % syncGradientFrequency ~= 0 then
            -- If no communication this round, get back on the default stream
            cutorch.setStream(mpi.async.streams.defaultStream)
            return res
         end
         if not m.parameters then return end
         local p, g = m:parameters()
         if not g then
            return res
         end
         for _, h in ipairs(backwardSynchronizationImpl(g, m)) do
            table.insert(mpicache.asyncDetails.handles, h)
         end
         return res
      end

      m.backward = asyncMPIBackward
   end


   -- BlockSequential backward is done via iterative backwardStep calls which return
   -- the parameters on which to synchronize.
   local function registerBlockSequentialAsyncMPIBackward(m)
      assert(torch.isTypeOf(m, 'nn.BlockSequential'))

      local function asyncMPIBackward(layer, input, gradOutput, scale)
         mpicache.asyncDetails.counter = mpicache.asyncDetails.counter + 1
         local res, param, gradient = nil, nil, nil
         while gradOutput do
            res = gradOutput
            gradOutput, param, gradient = layer:backwardStep(input, gradOutput, scale)
            if mpicache.asyncDetails.counter % syncGradientFrequency == 0 then
               -- Communicate this round (wrap gradient in a table)
               local handles = backwardSynchronizationImpl({gradient}, m)
               for _, h in ipairs(handles) do
                  table.insert(mpicache.asyncDetails.handles, h)
               end
            end
         end
         return res
      end

      m.backward = asyncMPIBackward
   end


   -- Dispatch to the proper backward registering function
   if torch.isTypeOf(net, 'nn.BlockSequential') then
      registerBlockSequentialAsyncMPIBackward(net)
   else
      if torch.isTypeOf(net, 'nn.Container') then
         local firstModule = true
         for _, m in ipairs(net:listModules()) do
            -- Freaking listModules also lists 'self', filter it out!
            if m ~= net then
               registerAsyncMPIBackward(m, firstModule)
               firstModule = false
            end
         end
      elseif net.parameters then
         registerAsyncMPIBackward(net)
      end
   end


   -- Synchronizing gradients in the context of async just means waiting for the
   -- pipelined collectives to finish.
   M.async.synchronizeGradients = function(net)
      for i = #mpicache.asyncDetails.handles, 1, -1 do
         mpi.syncHandle(mpicache.asyncDetails.handles[i])
      end
      mpicache.asyncDetails.handles = {}
   end
end

backwardSynchronizationImpl = function(tensors, m)
   if mpi.withCuda then
      -- assert we are on the right stream
      assert(cutorch.getStream() == mpi.async.streams.defaultStream)
      -- Local collective must wait for compute
      -- (i.e. current stream at this point)
      cutorch.streamWaitFor(
         mpi.async.streams.localCollectiveStream, { cutorch.getStream() })
      cutorch.setStream(mpi.async.streams.localCollectiveStream)
   end

   -- Allreduce return streams we need to sync on
   local handles = {}
   for i, gw in ipairs(tensors) do
      if not torch.isTensor(gw) then
         error('Not a tensor ' .. torch.type(gw) ..
                  '\n in module\n' .. tostring(m))
      end
      local allreduceTensor = selectCollective(gw, 'async', 'allreduceTensor')
      local handle = allreduceTensor(gw)
      table.insert(handles, handle)
   end

   if mpi.withCuda then
      cutorch.setStream(mpi.async.streams.defaultStream)
   end
   return handles
end


return M
