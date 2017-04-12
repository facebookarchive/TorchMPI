--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local cmd = torch.CmdLine()
cmd:option('-benchmark', false, 'skip correctness check and benchmark performance instead')
cmd:option('-cartesian', false, 'use cartesian or tree communicator')
cmd:option('-tests', 'all', 'Options: all | allselector | basic | p2p | nccl | gloo')
cmd:option('-processor', 'both', 'Options: gpu | cpu | both')
cmd:option('-execution', 'both', 'Options: sync | async | both: Dispatch collectives asynchronously and wait for handle before checking')
cmd:option('-storage', 'both', 'Options: inplace | copy | both: run inPlace or not')
cmd:option('-hierarchical', 'true', 'Use hierarchical collectives (true) or flat collectives (false)')
cmd:option('-staged', 'false', 'Use staged collectives (true) or flat collectives (false)')
cmd:option('-numBuffers', 3, 'Number of buffers to use for cpu or gpu collectives')
cmd:option('-minBufferSize', bit.lshift(1, 17), "Minimum buffer size for cpu and gpu collectives")
cmd:option('-maxBufferSize', bit.lshift(1, 20), "Maximum buffer size for cpu and gpu collectives")
cmd:option('-maxSizeForTreeBasedBroadcast', bit.lshift(1, 22), "Maximum size to use tree-based broadcast")
cmd:option('-collective', 'all', 'Run a single or all collectives. Options: all | broadcast | reduce | allreduce | sendreceivenext | allgather')
cmd:option('-type', 'float', 'Type of tensor to test. Option: char | int | long | float | double | all')
cmd:option('-uppersizepow', 23, 'Upper bound for exponent of tensor size, i.e. x for 2^x')

local config = cmd:parse(arg)
assert(config.tests == 'all' or config.tests == 'allselector' or
       config.tests == 'basic' or config.tests == 'p2p' or
       config.tests == 'nccl' or config.tests == 'gloo')
assert(config.collective == 'all' or config.collective == 'broadcast' or
       config.collective == 'reduce' or config.collective == 'allreduce' or
       config.collective == 'sendreceivenext' or config.collective == 'allgather')
assert(config.type == 'char' or config.type == 'int' or
       config.type == 'long' or config.type == 'float' or
       config.type == 'double' or config.type == 'all')

local gpuTable = nil
if config.processor == 'gpu' then
   gpuTable = {true}
elseif config.processor == 'cpu' then
   gpuTable = {false}
elseif config.processor == 'both' then
   gpuTable = {false, true}
else
   error("Illegal processor option: " .. config.processor)
end

local executionTable = nil
if config.execution == 'async' then
   executionTable = {true}
elseif config.execution == 'sync' then
   executionTable = {false}
elseif config.execution == 'both' then
   executionTable = {false, true}
else
   error("Illegal execution option: " .. config.execution)
end

local typeTable = nil
if config.type == 'all' then
   typeTable = {'char', 'int', 'long', 'float', 'double'}
else
   typeTable = {config.type}
end

local storageTable = nil
if config.storage == 'inplace' then
   storageTable = {true}
elseif config.storage == 'copy' then
   storageTable = {false}
elseif config.storage == 'both' then
   storageTable = {false, true}
else
   error("Illegal storage option: " .. config.storage)
end

config.check = not config.benchmark
if config.tests == 'nccl' then
   assert(config.processor == "gpu",
     'This test must be ran with GPUs, please specify -processor gpu.')
end

local nSkip = config.benchmark and 10 or 0
local nRuns = config.benchmark and 10 + nSkip or 1

-- If using GPUs, set the GPU before initializing MPI
local mpi = require('torchmpi')
mpi.start{withCuda = (config.processor ~= "cpu"),
          withIPCGroups = (config.tests ~= 'nccl'),
          withCartesianCommunicator = config.cartesian}

if config.hierarchical == 'true' then
  mpi.C.torchmpi_set_hierarchical_collectives()
else
  mpi.C.torchmpi_set_flat_collectives()
end

if config.staged == 'true' then
  mpi.C.torchmpi_set_staged_collectives()
else
  mpi.C.torchmpi_set_direct_collectives()
end

mpi.C.torchmpi_set_num_buffers_per_cpu_collective(config.numBuffers)
mpi.C.torchmpi_set_num_buffers_per_gpu_collective(config.numBuffers)

mpi.C.torchmpi_set_min_buffer_size_per_cpu_collective(config.minBufferSize)
mpi.C.torchmpi_set_min_buffer_size_per_gpu_collective(config.minBufferSize)

mpi.C.torchmpi_set_max_buffer_size_per_cpu_collective(config.maxBufferSize)
mpi.C.torchmpi_set_max_buffer_size_per_gpu_collective(config.maxBufferSize)

mpi.C.torchmpi_set_broadcast_size_cpu_tree_based(config.maxSizeForTreeBasedBroadcast)
mpi.C.torchmpi_set_broadcast_size_gpu_tree_based(config.maxSizeForTreeBasedBroadcast)

local collectiveAvailabilityCPU = mpi.collectiveAvailability(true, false)
local collectiveAvailabilityGPU = mpi.collectiveAvailability(false, true)
local tester = require('torchmpi.tester')
local asyncTimer = torch.Timer()

local function getCollectives()
   if config.tests == "allselector" then
      local sel = mpi.collectiveSelector
      sel = config.gpu and sel.gpu or sel.cpu
      sel = config.singlenode and sel.singlenode or sel.multinode
      sel = config.async and sel.async or sel.sync
      return sel
   else
      local sel = mpi
      sel = config.async and sel.async or sel
      if config.nccl then
         sel = mpi.hasNCCL and sel.nccl or nil
      end
      if config.gloo then
         if not config.gpu then
            sel = mpi.hasGloo and sel.gloo or nil
         else
            sel = mpi.hasGlooCuda and sel.gloo or nil
         end
      end
      if sel ~= nil then
        sel = config.p2p and sel.p2p or sel
      end

      return sel
   end
end

local function collectiveAvailable(ns, collective)
   if config.tests == "all" then
      local funcname = "MPI"
         .. (config.async and ".async" or "")
         .. (config.nccl and ".nccl" or "")
         .. (config.gloo and ".gloo" or "")
         .. (config.p2p and ".p2p." or ".")
         .. collective
      local availability = config.gpu and
         collectiveAvailabilityGPU or collectiveAvailabilityCPU
      local func = availability:match(funcname .. '[^\n]+')
      if func:match('unavailable') then
         return 'UNAVAILABLE'
      elseif func:match('unimplemented') then
         return 'NYI'
      end
   end
   return 'available'
end

local tests = {}

-------------------------------- reduce --------------------------------
tests.reduce = {}
tests.reduce.test = function(input, output, firstRun)
   -- Output must be zeroed explicitly to get proper results,
   -- only when out of place
   if input ~= output then output:zero() end

   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "reduceTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.reduceTensor(0, input, output)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async reduce launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.reduce.check = function(input, output)
   if mpi.rank() == 0 then
      local val = (mpi.size() * (mpi.size() - 1)) / 2
      local min, max = output:min(), output:max()
      assert(min == val, ('%f vs expected %f'):format(min, val))
      assert(max == val, ('%f vs expected %f'):format(max, val))
   end
end

-- Assumes a pipelined implementation of reduce
tests.reduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * elemSize) / 1e9
end

-------------------------------- broadcast --------------------------------
tests.broadcast = {}

tests.broadcast.test = function(input, output, firstRun)
   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "broadcastTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.broadcastTensor(mpi.size() - 1, input)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async broadcast launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.broadcast.check = function(input)
   -- only 1 tensor, no input/output distinction
   -- 0-based
   local val = mpi.size() - 1
   local min, max = input:min(), input:max()
   if min ~= val or max ~= val then
      error(('[%d/%d] %f vs expected %f %s (size: %d)\n'):format(
            mpi.rank(), mpi.size(), min, val, input:data(), input:nElement()))
   end
end

-- Assumes a pipelined implementation of broadcast
tests.broadcast.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * elemSize) / 1e9
end

-------------------------------- allreduce --------------------------------
tests.allreduce = {}

tests.allreduce.test = function(input, output, firstRun)
   -- Output must be zeroed explicitly to get proper results, only when out of place
   if input ~= output then output:zero() end

   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "allreduceTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.allreduceTensor(input, output)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async allreduce launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.allreduce.check = function(input, output, inputClone)
   local val = (mpi.size() * (mpi.size() - 1)) / 2
   local min, max = output:min(), output:max()
   if min ~= val or max ~= val then
      error(('[%d/%d] %f-%f vs expected %f (size %d)\n'):format(
            mpi:rank(), mpi:size(), min, max, val, output:nElement()))
   end

   -- check inPlace didn't write to input (and char:abs() seems broken)
   if not config.inPlace and config.type ~= 'char' then
      assert((input - inputClone):abs():max() == 0,
         "input changed after non inplace collective")
   end
end

-- Assumes a chunked-ring-based implementation of allreduce
-- (i.e. 1 roundtrip of the whole data through slowest wire to saturate BW)
tests.allreduce.communicationVolumeGB = function(input)
   local elemSize = 4
   return (2 * input:nElement() * elemSize * (mpi.size() - 1) / mpi.size()) / 1e9
end

-------------------------------- sendreceivenext -------------------------------
tests.sendreceivenext = {}

local dist = 1
tests.sendreceivenext.dist = math.min(dist, mpi.size() - 1)
tests.sendreceivenext.test = function(input, output, firstRun)
   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "sendreceiveTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.sendreceiveTensor(
      input,
      (mpi.rank() - tests.sendreceivenext.dist) % mpi.size(),
      (mpi.rank() + tests.sendreceivenext.dist) % mpi.size())

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async sendreceivenext launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement())
         )
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.sendreceivenext.check = function(input, output)
   output:copy(input)
   local val = (mpi.rank() - tests.sendreceivenext.dist) % mpi.size()
   local min, max = output:min(), output:max()
   assert(min == val)
   assert(max == val)
end

-- Pure point-to-point, 1 hop
tests.sendreceivenext.communicationVolumeGB = function(input)
   local elemSize = 4
   return input:nElement() * elemSize / 1e9
end
-------------------------------- allgather --------------------------------
tests.allgather = {}

tests.allgather.startsize = function(size)
   return math.max(0, math.ceil(size / mpi.size() - (mpi.size() + 1) / 2)) + 1
end

tests.allgather.outputsize = function(start_size)
   local end_size = start_size + (mpi.size() - 1)
   return end_size * (end_size + 1)  / 2  - start_size * (start_size - 1) / 2
end

tests.allgather.generate = function(size)
   -- pick an input size that increases by 1 for each rank that in total
   -- (output size) minimally exceeds size parameter, e.g.
   -- if size == 10 and mpi.size() == 3, then sizes are {3,4,5}
   local start_size = tests.allgather.startsize(size)
   local isize = start_size + mpi.rank()
   local input = torch[tester.tensorType(config.type, config.gpu)](isize)
   input:fill(mpi.rank())

   local osize = tests.allgather.outputsize(start_size)
   osize = config.inPlace and osize or osize / 2
   -- override meaning of inplace, because it doesn't make much sense for
   -- allgather.  Instead have it mean if the output needs to be ReAlloced.
   local output = torch[tester.tensorType(config.type, config.gpu)](osize)

   return input, output
end

tests.allgather.test = function(input, output, firstRun)
   local ns = getCollectives()
   local availability = collectiveAvailable(ns, "allgatherTensor")
   if availability ~= 'available' then
      return availability -- exit test loop
   end

   if config.async then
      asyncTimer:reset()
   end

   local handle = ns.allgatherTensor(input, output)

   if config.async and not firstRun then
      asyncTimer:stop()
      if asyncTimer:time().real >= 5e-5 then
         print(string.format(
            'Warning: Async allreduce launch took %f (expected < %f) for size %d',
            asyncTimer:time().real, 5e-5, input:nElement()))
      end
   end

   return handle
end

-- Careful, precision counts here once we reach a certain size
tests.allgather.check = function(input, output, inputClone)
   local min_val = 0
   local max_val = mpi.size() - 1
   local start_size = input:nElement() - mpi.rank()
   local size_val = tests.allgather.outputsize(start_size)
   local min, max = output:min(), output:max()
   if min ~= min_val or max ~= max_val then
      error(('[%d/%d] %f-%f vs expected %f-%f (size %d) vs expected size %d\n'):format(
            mpi:rank(), mpi:size(), min, max, min_val, max_val,
            output:nElement(), size_val))
   end

   -- check inPlace didn't write to input (and char:abs() seems broken)
   if not config.inPlace and config.type ~= 'char' then
      assert((input - inputClone):abs():max() == 0,
         "input changed after non inplace collective")
   end

   -- sample from each region
   local pos = 1
   for i=0,mpi.size()-1 do
      if (output[pos + i] ~= i) then
         error(('expected %d but got %d at position %d\n'):format(
               i, output[pos + i + 1], pos + mpi.size()))
      end
      pos = pos + start_size + i
   end
end

-- Assumes a pipelined implementation of each gather (equivalent to broadcast above)
tests.allgather.communicationVolumeGB = function(input)
   local elemSize = 4
   return (input:nElement() * (mpi.size() -1 )* elemSize) / 1e9
end
-------------------------------- barrier --------------------------------
tests.mpiBarrier = {}
tests.mpiBarrier.test = function()
   mpi.barrier()
end
tests.mpiBarrier.check = function() end
tests.mpiBarrier.communicationVolumeGB = function() return 0 end

-------------------------------- custom barrier --------------------------------
local ffi = require('ffi')
ffi.cdef [[
   void customBarrier();
]]
tests.customBarrier = {}
tests.customBarrier.test = function()
   ffi.C.customBarrier()
end
tests.customBarrier.check = function() end
tests.customBarrier.communicationVolumeGB = function() return 0 end

-------------------------------- Start tests --------------------------------\
local function setImplemented()
   if config.tests == "all" or config.tests == "allselector" then
      tests.broadcast.implemented = true
      tests.reduce.implemented = true
      tests.allreduce.implemented = true
      tests.sendreceivenext.implemented = true
      tests.allgather.implemented =
         not (config.async or config.p2p or config.gloo)
   elseif config.tests == "basic" then
      -- No async sendreceivenext
      tests.sendreceivenext.implemented = not config.async
      tests.mpiBarrier.implemented = true
      -- Disable because it deadlocks on multi-machines
      tests.customBarrier.implemented = false
      tests.broadcast.implemented = true
      -- No async sendreceivenextGPU reduce
      tests.reduce.implemented = not (config.async and config.gpu)
      tests.allreduce.implemented = true
      tests.allgather.implemented = not config.async
   elseif config.tests == "p2p" then
      tests.broadcast.implemented = true
      tests.allreduce.implemented = true
   elseif config.tests == "nccl" then
      tests.broadcast.implemented = mpi.hasNCCL
      tests.reduce.implemented = mpi.hasNCCL
      tests.allreduce.implemented = mpi.hasNCCL
   elseif config.tests == "gloo" then
      local implemented =
         (not mpi.gpu and mpi.hasGloo and config.inPlace) or
         (mpi.gpu and mpi.hasGlooCuda and config.inPlace)
      tests.broadcast.implemented = implemented
      tests.allreduce.implemented = implemented
   end

   local function enabled(name)
      return config.collective == "all" or config.collective == name
   end
   tests.sendreceivenext.implemented =
      tests.sendreceivenext.implemented and enabled("sendreceivenext")
   tests.mpiBarrier.implemented =
      tests.mpiBarrier.implemented and enabled("mpiBarrier")
   tests.customBarrier.implemented =
      tests.customBarrier.implemented and enabled("customBarrier")
   tests.broadcast.implemented =
      tests.broadcast.implemented and enabled("broadcast")
   tests.reduce.implemented =
      tests.reduce.implemented and enabled("reduce")
   tests.allreduce.implemented =
      tests.allreduce.implemented and enabled("allreduce")
   tests.allgather.implemented =
      tests.allgather.implemented and enabled("allgather")
end

local function ncclTable(gpu)
   if config.tests == "all" and gpu then
      return {false, true}
   elseif config.tests == "all" and not gpu then
      return {false}
   elseif config.tests == "nccl" then
      return {true}
   else
      return {false}
   end
end

local function glooTable(inPlace, nccl, p2p)
   if config.tests == "all" then
      return (inPlace and not nccl and not p2p) and {false, true} or {false}
   elseif config.tests == "gloo" then
      return {true}
   else
      return {false}
   end
end

if config.tests == "allselector" then
   for _, async in ipairs(executionTable) do
      for _, gpu in ipairs(gpuTable) do
         for _, inPlace in ipairs(storageTable) do
            for _, singlenode in ipairs({false, true}) do
               for _, type in ipairs(typeTable) do
                  config.async = async
                  config.gpu = gpu
                  config.inPlace = inPlace
                  config.singlenode = singlenode
                  config.type = type
                  setImplemented()
                  tester.runOneConfig(tests, nRuns, nSkip, config)
               end
            end
         end
      end
   end
else
   for _, async in ipairs(executionTable) do
      for _, gpu in ipairs(gpuTable) do
         for _, inPlace in ipairs(storageTable) do
            local p2pTable = config.tests == "all" and {false, true}
                             or config.tests == "p2p" and {true} or {false}
            for _, p2p in ipairs(p2pTable) do
               for _, nccl in ipairs(ncclTable(gpu)) do
                  for _, gloo in ipairs(glooTable(inPlace, nccl, p2p)) do
                     for _, type in ipairs(typeTable) do
                        config.async = async
                        config.gpu = gpu
                        config.inPlace = inPlace
                        config.nccl = nccl
                        config.gloo = gloo
                        config.p2p = p2p
                        config.type = type
                        setImplemented()
                        tester.runOneConfig(tests, nRuns, nSkip, config)
                     end
                   end
               end
            end
         end
      end
   end
end

mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
