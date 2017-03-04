--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require 'nn'
local mpi      = require 'torchmpi'
local mpinn    = require 'torchmpi.nn'
local tnt      = require 'torchnet'
local tablex   = require 'pl.tablex'
local argcheck = require 'argcheck'

local AREngine, SGDEngine =
   torch.class('tnt.AllReduceSGDEngine', 'tnt.SGDEngine', tnt)

AREngine.__init = argcheck{
   {name="self",   type="tnt.AllReduceSGDEngine"},
   {name="usegpu", type="boolean", default = false},
   {name="async", type="boolean", default = false},
   {name="devicesync", type="boolean", default = true},
   {name="dynamicnetwork", type="boolean", default = true},
   {name="debug", type="boolean", default = false},
   call = function(self, usegpu, async, devicesync, dynamicnetwork, debug)
      assert(mpi, 'only supported when running with MPI')
      SGDEngine.__init(self)
      self.usegpu = usegpu
      self.async = async
      self.devicesync = devicesync
      self.dynamicnetwork = dynamicnetwork
      self.debug = debug
      self.registered = false
   end
}

local function nvprof(name, ...)
   if os.getenv("NVPROF") then
      local args = {...}
      local state = args[1]

      local ffi = require('ffi')
      ffi.cdef [[
               void exit(int);
               void cudaDeviceReset();
               void cudaProfilerStart();
               void cudaProfilerStop();
            ]]
      if state.t == 3 and name == 'onSample' then
         mpi.barrier()
         cutorch.synchronize()
         ffi.C.cudaProfilerStart()
      elseif state.t == 8 and name == 'onSample' then
         mpi.barrier()
         cutorch.synchronize()
         ffi.C.cudaProfilerStop()
         ffi.C.cudaDeviceReset()
      elseif state.t >= 8 then
         return
      end
   end
end

AREngine.train = argcheck{
   {name="self", type="tnt.AllReduceSGDEngine"},
   {name="network", type="nn.Module"},
   {name="criterion", type="nn.Criterion"},
   {name="iterator", type="tnt.DatasetIterator"},
   {name="lr", type="number"},
   {name="lrcriterion", type="number", defaulta="lr"},
   {name="maxepoch", type="number", default=1000},
   call = function(
      self, network, criterion, iterator, lr, lrcriterion, maxepoch)

      -- get hooks metatable and store copy:
      local mt = getmetatable(self.hooks)
      if not mt then error('hooks metatable not found') end
      local mtclone = tablex.copy(mt)

      -- composes hook with MPI stuff:
      mt.__call = function(hooks, name, ...)

         nvprof(name, ...)

         -- get input arguments to hook:
         local args = {...}
         local state = args[1]

         -- implementations of hooks:
         if name == 'onSample' then
            if self.async then
               -- Register the async backward pass
               -- If you are using GPUs you are responsible for having called
               -- torchmpi.async.initStreams(true)
               -- Do this right before porential CPU synchronization so it is
               -- hidden by compute
               -- TODO: detect if network changed and needs to be re-registered
               if not self.registered then
                  mpinn.async.registerAsyncMPIBackward(state.network)
                  mpinn.async.registerAsyncMPIBackward(state.criterion)
                  if not self.dynamicnetwork then
                     -- a dynamic network is never considered 'registered'
                     self.registered = true
                  end
               end
            end
            -- Synchronize after sample and not at the end of minibatch.
            -- Synchronizing at end of minibatch too easily synchronizes
            --    GPU -> CPU -> data loading
            if self.devicesync then
               mpi.barrier()
               if self.usegpu then cutorch.synchronize() end
            end
            if self.debug then
               mpinn.checkWithAllreduce(state.network)
               mpinn.checkWithAllreduce(state.criterion)
            end
         elseif name == 'onBackwardCriterion' then
            if self.devicesync then
               -- Synchronize right after forward pass, not strictly necessary
               mpi.barrier()
               if self.usegpu then cutorch.synchronize() end
            end
            if iterator.prefetch then iterator:prefetch() end
         elseif name == 'onBackward' then
            if mpi.size() > 1 then
               local impl = self.async and mpinn.async or mpinn
               impl.synchronizeGradients(state.criterion)
               impl.synchronizeGradients(state.network)
            end
         end

         return hooks[name](...)
      end

      -- run trainer and restore original metatable:
      setmetatable(self.hooks, mt)

      if mpi.size() > 1 then
         -- Initial synchronize of parameters (one shot, synchronous)
         mpinn.synchronizeParameters(network)
         mpinn.synchronizeParameters(criterion)
      end

      SGDEngine.train(
         self, network, criterion, iterator, lr, lrcriterion, maxepoch)
      setmetatable(self.hooks, mtclone)
   end
}
