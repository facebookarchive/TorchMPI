--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')
local MPI = require("torchmpi.env")
local argcheck = require 'argcheck'
local types = require("torchmpi.types")
local wrap = require('torchmpi.wrap')

MPI.async = {}
MPI.async.streams = nil
MPI.async.initStreams = function(async)
   if not MPI.async.streams then
      MPI.async.streams = {}
      cutorch.reserveStreams(5, async)
      MPI.async.streams.defaultStream = 0
      MPI.async.streams.copyToGPUStream = 1
      MPI.async.streams.copyToCPUStream = 2
      MPI.async.streams.localCollectiveStream = 3
      MPI.async.streams.globalCollectiveStream = 4
   end
end

local setupNCCL, initPerNodeCommunicators, configureCollectiveSelector

MPI.start = argcheck {
   { name = "withCuda", type = "boolean" },
   { name = "withIPCGroups", type = "boolean", default = true },
   { name = "customCommunicatorInit", type = "function", opt = true  },
   { name = "withCartesianCommunicator", type = "boolean", default = false },
   { name = "collectiveCommunicator", type = "function", opt = true },
   call = function(withCuda, withIPCGroups, customCommunicatorInit, withCartesianCommunicator, collectiveCommunicator)
      MPI.ipcGroups = (ipcGroups == nil) and true or ipcGroups

      local function getHostname()
         local f = io.popen("/bin/hostname")
         local hostname = f:read("*a") or ""
         f:close()
         hostname = string.gsub(hostname, "\n$", "")
         return hostname
      end

      -- anything that forks processes must happen before torchmpi_start
      MPI.hostName = getHostname()
      MPI.withCuda = withCuda

      require('torchmpi.ffi')(MPI.withCuda)
      MPI.hasNCCL = (MPI.C.torchmpi_has_nccl() ~= 0) and true or false
      if MPI.hasNCCL then setupNCCL() end

      MPI.hasGloo = (MPI.C.torchmpi_has_gloo() ~= 0) and true or false
      if MPI.hasGloo then setupGloo() end

      MPI.hasGlooCuda = (MPI.C.torchmpi_has_gloo_cuda() ~= 0) and true or false

      if not withCartesianCommunicator then
         MPI.C.torchmpi_set_tree_communicator()
      else
         MPI.C.torchmpi_set_cartesian_communicator()
      end

      MPI.C.torchmpi_start()

      if MPI.withCuda then
         assert(
            os.getenv('OMPI_COMM_WORLD_LOCAL_RANK') or
               os.getenv('MV2_COMM_WORLD_LOCAL_RANK'),
            'OMPI_COMM_WORLD_LOCAL_RANK or MV2_COMM_WORLD_LOCAL_RANK env var must be defined')
         MPI.supportsCuda = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK') or
            os.getenv('MV2_COMM_WORLD_LOCAL_RANK')
         MPI.device =
            tonumber(
               os.getenv('OMPI_COMM_WORLD_LOCAL_RANK') or
                  os.getenv('MV2_COMM_WORLD_LOCAL_RANK')
            ) % cutorch.getDeviceCount() + 1
         cutorch.setDevice(MPI.device)
      end

      if customCommunicatorInit then
         MPI.numCommunicators =
            assert(customCommunicatorInit(),
                   'customCommunicatorInit should return the communicator hierarchy\'s depth')
         -- Principle of least surprise, if there is a custom communicator
         -- it is assumed the communicatrors for collectives are rooted under it
         MPI.C.torchmpi_set_communicator(MPI.numCommunicators)
      end

      if not withCollectiveCommunicator then
         initPerNodeCommunicators()
      else
         collectiveCommunicator()
      end
      configureCollectiveSelector()
   end
}

MPI.stop = function()
   MPI.C.torchmpi_stop()
end

MPI.barrier = function()
   MPI.C.torchmpi_barrier()
end

MPI.size = function()
   return MPI.C.torchmpi_size()
end

MPI.rank = function()
   return MPI.C.torchmpi_rank()
end

local ffi = require("ffi")

MPI.communicatorNames = function()
   return ffi.string(MPI.C.torchmpi_communicator_names())
end

-- Scalar low-latency operations
for _, v in ipairs(types.C) do
   MPI['allreduce_' .. v] =
     function(val) return MPI.C['torchmpi_allreduce_' .. v](val) end
   MPI['broadcast_' .. v] =
     function(rank, val) return MPI.C['torchmpi_broadcast_' .. v](val, rank) end
   MPI['reduce_' .. v] =
     function(rank, val) return MPI.C['torchmpi_reduce_' .. v](val, rank) end
   MPI['sendreceive_' .. v] =
     function(val) return MPI.C['torchmpi_sendreceive_' .. v](val) end
end


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
----------------------------- Regular collectives ------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--------------------- Synchronous collectives CPU or GPU -----------------------

MPI.broadcastTensor = function(rank, tensor)
   local fun = 'torchmpi_broadcast_TH'..torch.type(tensor):gsub('torch.', '')
   return MPI.syncHandle(wrap.executeMPICFun(fun, tensor, rank))
end

MPI.reduceTensor = function(rank, input, output)
   local fun = 'torchmpi_reduce_TH'..torch.type(input):gsub('torch.', '')
   return MPI.syncHandle(wrap.executeMPICFun2(fun, input, output or input, rank))
end

MPI.allreduceTensor = function(input, output)
   local fun = 'torchmpi_allreduce_TH'..torch.type(input):gsub('torch.', '')
   return MPI.syncHandle(wrap.executeMPICFun2(fun, input, output or input))
end

MPI.sendreceiveTensor = function(input, src, dst)
   local fun = 'torchmpi_sendreceive_TH'..torch.type(input):gsub('torch.', '')
   return MPI.syncHandle(wrap.executeMPICFun(fun, input, src, dst))
end

--------------------- Asynchronous collectives CPU or GPU ----------------------

MPI.syncHandle = function(handle)
   return MPI.C.torchmpi_synchronize_handle(handle)
end

-- Returns a handle on which to call MPI.syncHandle to ensure the
-- collective completed
MPI.async.broadcastTensor = function(rank, tensor)
   local fun = 'torchmpi_async_broadcast_TH'..torch.type(tensor):gsub('torch.', '')
   return wrap.executeMPICFun(fun, tensor, rank)
end

-- Returns a handle on which to call MPI.syncHandle to ensure the
-- collective completed
MPI.async.reduceTensor = function(rank, input, output)
   if torch.type(input):find('Cuda') then
      -- TODO: implement asynchronous reduce on GPU tensors if really needed
      -- local fun = 'torchmpi_reduce_TH'..torch.type(input):gsub('torch.', '')
      -- return wrap.executeMPICFun2(fun, input, output or input , rank)
      error('NYI')
   elseif not output or output:data() == input:data() then
      -- TODO: OpenMPI IAllreduce seems bugged ???
      return MPI.reduceTensor(rank, input)
   end
   local fun = 'torchmpi_async_reduce_TH'..torch.type(input):gsub('torch.', '')
   return wrap.executeMPICFun2(fun, input, output or input, rank)
end

-- Returns a handle on which to call MPI.syncHandle to ensure the
-- collective completed
MPI.async.allreduceTensor = function(input, output)
   local fun = 'torchmpi_async_allreduce_TH'..torch.type(input):gsub('torch.', '')
   return wrap.executeMPICFun2(fun, input, output or input)
end

-- TODO: implement asynchronous sendreceive on CPU/GPU tensors
MPI.async.sendreceiveTensor = nil --[[function(input, src, dst)
   local fun = 'torchmpi_sendreceive_TH'..torch.type(input):gsub('torch.', '')
   return wrap.executeMPICFun(fun, input, src, dst)
end
--]]



--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------- Custom P2P collectives -----------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

--------------------- Synchronous collectives CPU or GPU -----------------------
MPI.p2p = {}

MPI.p2p.broadcastTensor = function(rank, tensor)
   local fun = 'torchmpi_p2p_broadcast_TH'..
      torch.type(tensor):gsub('torch.', '')
   return MPI.syncHandle(wrap.executeMPICFun(fun, tensor, rank))
end

--TODO: Implement if really needed
MPI.p2p.reduceTensor = nil --[[ function(rank, input, output)
   return MPI.syncHandle(MPI.reduceTensor(rank, input, output))
end
--]]

MPI.p2p.allreduceTensor = function(input, output)
   local fun = 'torchmpi_p2p_allreduce_TH'..torch.type(input):gsub('torch.', '')
   return MPI.syncHandle(wrap.executeMPICFun2(fun, input, output or input))
end

--TODO: Implement if really needed
MPI.p2p.sendreceiveTensor = nil --[[ function(input, src, dst)
   return MPI.syncHandle(MPI.sendreceiveTensor(input, src, dst))
end
--]]

--------------------- Asynchronous collectives CPU or GPU ----------------------
MPI.async.p2p = {}

-- Returns a handle on which to call MPI.syncHandle to ensure the
-- collective completed
MPI.async.p2p.broadcastTensor = function(rank, tensor)
   local fun = 'torchmpi_async_p2p_broadcast_TH'
      ..torch.type(tensor):gsub('torch.', '')
   return wrap.executeMPICFun(fun, tensor, rank)
end

-- Returns a handle on which to call MPI.syncHandle to ensure the
-- collective completed
MPI.async.p2p.reduceTensor = function(rank, input, output)
   return MPI.async.reduceTensor(rank, input, output)
end

-- Runs the collective in a helper thread that synchronizes with a CUDA stream
-- Returns a handle on which to call MPI.syncHandle to ensure the
-- collective completed
MPI.async.p2p.allreduceTensor = function(input, output)
   local fun =
      'torchmpi_async_p2p_allreduce_TH'..torch.type(input):gsub('torch.', '')
   return wrap.executeMPIChunkedAllreduceCFun(fun, input, output or input)
end

--TODO: Implement if really needed
MPI.async.p2p.sendreceiveTensor = nil --[[function(input, src, dst)
   return MPI.sendreceiveTensor(input, src, dst)
end
--]]


--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
------------------------------ NCCL collectives --------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

setupNCCL = function()
   --------------------- Synchronous collectives GPU only -------------------------
   MPI.nccl = {}

   -- NCCL collectives wrappers return the stream on which they execute
   -- so we can wait for the collective to complete.
   MPI.nccl.allreduceTensor = function(input, output)
      local fun = 'torchmpi_nccl_allreduce_TH'..torch.type(input):gsub('torch.', '')
      return MPI.syncHandle(wrap.executeMPICFun2(fun, input, output or input))
   end

   MPI.nccl.broadcastTensor = function(rank, tensor)
      local fun = 'torchmpi_nccl_broadcast_TH'..torch.type(tensor):gsub('torch.', '')
      return MPI.syncHandle(wrap.executeMPICFun(fun, tensor, rank))
   end

   MPI.nccl.reduceTensor = function(rank, input, output)
      local fun = 'torchmpi_nccl_reduce_TH'..torch.type(input):gsub('torch.', '')
      return MPI.syncHandle(wrap.executeMPICFun2(fun, input, output or input, rank))
   end

   MPI.nccl.sendreceiveTensor = nil --[[function(input, src, dst)
      return MPI.nccl.sendreceiveTensor(input, src, dst)
      end
   --]]

   --------------------- Asynchronous collectives GPU only ------------------------
   MPI.async.nccl = {}

   -- NCCL collectives wrappers return the stream on which they execute
   -- so we can wait for the collective to complete.
   MPI.async.nccl.allreduceTensor = function(input, output)
      local fun = 'torchmpi_async_nccl_allreduce_TH'..torch.type(input):gsub('torch.', '')
      return wrap.executeMPICFun2(fun, input, output or input)
   end

   MPI.async.nccl.broadcastTensor = function(rank, tensor)
      local fun = 'torchmpi_async_nccl_broadcast_TH'..torch.type(tensor):gsub('torch.', '')
      return wrap.executeMPICFun(fun, tensor, rank)
   end

   MPI.async.nccl.reduceTensor = function(rank, input, output)
      local fun = 'torchmpi_async_nccl_reduce_TH'..torch.type(input):gsub('torch.', '')
      return wrap.executeMPICFun2(fun, input, output or input, rank)
   end

   MPI.async.nccl.sendreceiveTensor = nil --[[ function(input, src, dst)
      --TODO: Implement
      return MPI.sendreceiveTensor(input, src, dst)
      end
   --]]
end

setupGloo= function()
   --------------------- Synchronous collectives CPU only -------------------------
   MPI.gloo = {}

   MPI.gloo.broadcastTensor = function(rank, tensor)
      local fun = 'torchmpi_gloo_broadcast_TH'..torch.type(tensor):gsub('torch.', '')
      return MPI.syncHandle(wrap.executeMPICFun(fun, tensor, rank))
   end

   MPI.gloo.allreduceTensor = function(input, output)
      local fun = 'torchmpi_gloo_allreduce_TH'..torch.type(input):gsub('torch.', '')
      return MPI.syncHandle(wrap.executeMPICFun2(fun, input, output or input))
   end

   --------------------- Asynchronous collectives CPU only------------------------
   MPI.async.gloo = {}

   MPI.async.gloo.broadcastTensor = function(rank, tensor)
      local fun = 'torchmpi_async_gloo_broadcast_TH'..torch.type(tensor):gsub('torch.', '')
      return wrap.executeMPICFun(fun, tensor, rank)
   end

   MPI.async.gloo.allreduceTensor = function(input, output)
      local fun = 'torchmpi_async_gloo_allreduce_TH'..torch.type(input):gsub('torch.', '')
      return wrap.executeMPICFun2(fun, input, output or input)
   end
end

-- Performs sanity checks using MPI.allreduceTensor
-- The simple idea is that if distributed quantities are the same,
-- their mean by an allreduce should be the same modulo precision errors.
-- Very useful for debugging parameterservers once allreduce functional
-- correctness is established.
MPI.checkWithAllreduce = function(input, name, debugprint)
   local name = name or ''
   local absmean = torch.isTensor(input) and
      input:float():clone():abs():mean() or
      math.abs(input)
   local absmean_sync = MPI.allreduce_double(absmean) / MPI.size()
   local absvar = torch.isTensor(input) and
      input:float():clone():abs():var() or
      math.abs(input)
   local absvar_sync = MPI.allreduce_double(absvar) / MPI.size()
   if debugprint then
      for i = 0, MPI.size() - 1 do
         MPI.barrier()
      end
   end
   assert((absmean == 0 and absmean_sync == 0) or
      math.abs((absmean - absmean_sync) / absmean) < 1e-7,
      name .. ': absmean ' .. tostring(absmean) ..
         ' vs absmean_sync ' .. tostring(absmean_sync))
   assert((absvar == 0 and absvar_sync == 0) or
      math.abs((absvar - absvar_sync) / absvar) < 1e-7,
      name .. ': absvar ' .. tostring(absvar) ..
         ' vs absvar_sync ' .. tostring(absvar_sync))
end


---------------------------------- Helpers -------------------------------------

-- Sets up a 2-level communicator below the current communicator:
--   1. First-level is inter-(node, cudaIPC group)
--   2. Second-level is inter-(node, cudaIPC group)
-- where cudaIPC group is the list of GPUs that can talk to each other
-- directly via cudaIPC (a prerequisite is they dangle from the same socket).
-- In the case of CPU-only, the cudaIPC group is just an empty string.
-- Sets up a collective span of *2* levels.
--
-- The underlying cpp implementation adapts to the type of collective used and
-- the type of transport (TCP / RDMA).
--
-- In practice if you did not specify an initCommunicatorFun then all
-- processes will be involved in the collective.
-- On the contrary, say you decide to specify (a) coarse grained communicator(s)
-- for parameterserver behavior, then within each leaf you get an extra level
-- communicator. The last 2 levels form the unit of granularity for collectives
-- (i.e. synchronous SGD)
initPerNodeCommunicators = function()
   local numCommunicatorsOnEntry = MPI.numCommunicators or 0
   local accessible = ''

   -- Do we have a single ipcGroup in the machine or more ?
   if MPI.withCuda then
      MPI.singleIPCGroup = true
      for other = 1, cutorch.getDeviceCount() do
         local canAccess = cutorch.getPeerToPeerAccess(cutorch.getDevice(), other)
         if canAccess then
            accessible = accessible .. other .. '-'
         else
            MPI.singleIPCGroup = false
         end
      end
      if MPI.ipcGroup then
         accessible = accessible
      else
         accessible = ''
      end
   end

   -- Push 2 levels of communicators (node and cudap2p within node)
   MPI.numCommunicators =
      MPI.C.torchmpi_push_communicator(
         MPI.hostName .. ' cuda p2p group(' .. accessible .. ')')

   -- Set the collective span to 2 levels starting from numCommunicatorsOnEntry
   MPI.C.torchmpi_set_collective_span(
      numCommunicatorsOnEntry, numCommunicatorsOnEntry + 1)

   MPI.C.torchmpi_set_communicator(numCommunicatorsOnEntry)
   MPI.needInterNodeCollectives = (MPI.C.torchmpi_num_nodes_in_communicator() > 1)

   -- Come back to the original scope and synchronize
   MPI.C.torchmpi_set_communicator(numCommunicatorsOnEntry)
   MPI.C.torchmpi_barrier()

   --if MPI.rank() == 0 then
      print('[MPI] Using the following hierarchical communicators:')
      print('[MPI] Collective span: ',
         numCommunicatorsOnEntry .. ' - ' .. numCommunicatorsOnEntry + 1)
      print(MPI.communicatorNames())
   --end
end

configureCollectiveSelector = function()
   MPI.collectiveSelector = {
      cpu = {
         singlenode = {
            sync = {
               allreduceTensor = MPI.p2p.allreduceTensor or (MPI.gloo and MPI.gloo.allreduceTensor),
               broadcastTensor = MPI.broadcastTensor or (MPI.gloo and MPI.gloo.broadcastTensor),
               reduceTensor = MPI.reduceTensor,
               sendreceiveTensor = MPI.sendreceiveTensor,
            },
            async = {
               allreduceTensor = MPI.async.p2p.allreduceTensor or (MPI.gloo and MPI.async.gloo.allreduceTensor),
               broadcastTensor = MPI.async.broadcastTensor or (MPI.gloo and MPI.async.gloo.broadcastTensor),
               reduceTensor = MPI.reduceTensor,           -- OpenMPI-1.8  async version seems bugged ??
               sendreceiveTensor = MPI.sendreceiveTensor, -- no async version
            },
         },
         multinode = {
            sync = {
               allreduceTensor = MPI.p2p.allreduceTensor or (MPI.gloo and MPI.gloo.allreduceTensor),
               broadcastTensor = MPI.broadcastTensor or (MPI.gloo and MPI.gloo.broadcastTensor),
               reduceTensor = MPI.reduceTensor,
               sendreceiveTensor = MPI.sendreceiveTensor,
            },
            async = {
               allreduceTensor = MPI.async.p2p.allreduceTensor or (MPI.gloo and MPI.async.gloo.allreduceTensor),
               broadcastTensor = MPI.async.broadcastTensor or (MPI.gloo and MPI.gloo.broadcastTensor),
               reduceTensor = MPI.reduceTensor,           -- OpenMPI-1.8 async version seems bugged ??
               sendreceiveTensor = MPI.sendreceiveTensor, -- no async version
            },
         },
      },
      gpu = {
         singlenode = {
            sync = {
               allreduceTensor = MPI.ipcGroups and MPI.p2p.allreduceTensor
                  or (MPI.nccl and MPI.nccl.allreduceTensor)
                  or MPI.allreduceTensor
                  or (MPI.hasGlooCuda and MPI.gloo and MPI.gloo.allreduceTensor),
               -- Cross ipcGroups p2p.broadcast does not work, use stock MPI
               broadcastTensor = MPI.singleIPCGroup and MPI.p2p.broadcastTensor or
                  MPI.ipcGroups and MPI.broadcastTensor
                  or (MPI.nccl and MPI.nccl.broadcastTensor)
                  or MPI.broadcastTensor
                  or (MPI.hasGlooCuda and MPI.gloo and MPI.gloo.broadcastTensor),
               reduceTensor = MPI.nccl and MPI.nccl.reduceTensor or MPI.reduceTensor,
               sendreceiveTensor = MPI.sendreceiveTensor,
            },
            async = {
               allreduceTensor = MPI.ipcGroups and MPI.async.p2p.allreduceTensor
                  or (MPI.nccl and MPI.async.nccl.allreduceTensor)
                  or MPI.async.allreduceTensor
                  or (MPI.gloo and MPI.async.gloo.allreduceTensor),
               broadcastTensor = MPI.ipcGroups and MPI.async.p2p.broadcastTensor
                  or (MPI.nccl and MPI.async.nccl.broadcastTensor)
                  or MPI.async.broadcastTensor
                  or (MPI.hasGlooCuda and MPI.gloo and MPI.async.gloo.broadcastTensor),
               reduceTensor = MPI.nccl and MPI.async.nccl.reduceTensor or MPI.reduceTensor, -- OpenMPI-1.8 async version seems bugged ??
               sendreceiveTensor = MPI.sendreceiveTensor, -- no async version
            },
         },
         multinode = {
            sync = {
               allreduceTensor = MPI.ipcGroups and MPI.p2p.allreduceTensor
                  or (MPI.nccl and MPI.nccl.allreduceTensor)
                  or MPI.allreduceTensor
                  or (MPI.hasGlooCuda and MPI.gloo and MPI.gloo.allreduceTensor),
               broadcastTensor = MPI.broadcastTensor or (MPI.gloo and MPI.gloo.broadcastTensor),
               reduceTensor = MPI.reduceTensor,
               sendreceiveTensor = MPI.sendreceiveTensor,
            },
            async = {
               allreduceTensor = MPI.ipcGroups and MPI.async.p2p.allreduceTensor
                  or (MPI.nccl and MPI.async.nccl.allreduceTensor)
                  or MPI.async.allreduceTensor
                  or (MPI.hasGlooCuda and MPI.gloo and MPI.async.gloo.allreduceTensor),
               broadcastTensor = MPI.async.broadcastTensor
                   or (MPI.hasGlooCuda and MPI.gloo and MPI.async.gloo.broadcastTensor),
               reduceTensor = MPI.reduceTensor,           -- OpenMPI-1.8 async version seems bugged ??
               sendreceiveTensor = MPI.sendreceiveTensor, -- no async version
            },
         },
      },
   }
end

function MPI.collectiveAvailability(cpu, gpu)
   -- use collectiveSelector as a proxy for having been inited.
   if cpu == nil then cpu = true end
   if gpu == nil then gpu = true end
   if MPI.collectiveSelector == nil then
      return nil
   end

   local str = ""
   local cpugpu = {}
   if cpu then table.insert(cpugpu, false) end
   if gpu then table.insert(cpugpu, true) end
   for _, gpu in ipairs(cpugpu) do
      if not gpu then
         str = str .. "cpu = {\n"
      else
         str = str .. "gpu = {\n"
      end
      for _, async in ipairs({false, true}) do
         for _, nccl in ipairs({false, true}) do
            for _, gloo in ipairs(nccl and {false} or {false, true}) do
               for _, p2p in ipairs(gloo and {false} or {false, true}) do
                  for _, collective in ipairs({"broadcast", "reduce", "allreduce", "sendreceive"}) do
                     if gpu or not nccl then -- cpu + nccl not valid
                        local funcname = "MPI" .. (async and ".async" or "")
                            .. (nccl and ".nccl" or "" ) .. (gloo and ".gloo" or "")
                            .. (p2p and ".p2p." or ".") .. collective .. "Tensor"

                        local func = MPI
                        func = async and func.async or func
                        if nccl then
                           func = MPI.hasNCCL and func.nccl or nil
                        end
                        if gloo then
                           if not gpu then
                              func = MPI.hasGloo and func.gloo or nil
                           else
                              func = MPI.hasGlooCuda and func.gloo or nil
                           end
                        end
                        if func ~= nil then
                           func = p2p and func.p2p or func
                           func = func[collective .. "Tensor"]
                        end
                        local val = func ~= nil and "available" or "unimplemented"

                        -- differentiate between unimplemented and unavailable
                        if func == nil and nccl and not MPI.hasNCCL then
                           val = collective == "sendreceive" and "unimplemented" or "unavailable"
                        end

                        if func == nil and gloo and ((not gpu and not MPI.hasGloo) or (gpu and not MPI.hasGlooCuda)) then
                           val = (collective == "sendreceive" or collective == "reduce") and "unimplemented" or "unavailable"
                        end

                        -- special cases
                        if gpu and async and not nccl and collective == "reduce" then
                           val = "unimplemented"
                        end
                        str = str .. ('\t%-35s \t->\t %s\n'):format(funcname, val)
                     end
                  end
               end
            end
         end
      end
      str = str .. "}\n"
   end

   return str
end

MPI.collectiveSelectorToString = function(cpuSel, nodeSel, asyncSel, collSel)
   assert(not cpuSel or cpuSel == 'cpu' or cpuSel == 'gpu', 'Invalid cpuSelector string ' .. tostring(cpuSel))
   assert(not nodeSel or nodeSel == 'singlenode' or nodeSel == 'multinode', 'Invalid nodeSelector string ' .. tostring(nodeSel))
   assert(not asyncSel or asyncSel == 'sync' or asyncSel == 'async', 'Invalid asyncSelector string ' .. tostring(asyncSel))
   assert(not collSel or collSel == 'allreduceTensor' or collSel == 'broadcastTensor' or collSel == 'reduceTensor' or collSel == 'sendreceiveTensor',
          'Invalid collectiveSelector string ' .. tostring(collSel))
   local str = ''
   local fun2string = {}
   for _, coll in ipairs({'allreduceTensor', 'broadcastTensor', 'reduceTensor', 'sendreceiveTensor'}) do
      fun2string[MPI[coll]] = 'MPI.'..coll
      if MPI.async[coll] then fun2string[MPI.async[coll]] = 'MPI.async.'..coll end
      if MPI.nccl then
         if MPI.nccl[coll]then fun2string[MPI.nccl[coll]] = 'MPI.nccl.'..coll end
         if MPI.async.nccl[coll] then fun2string[MPI.async.nccl[coll]] = 'MPI.async.nccl.'..coll end
      end
      if MPI.p2p[coll] then fun2string[MPI.p2p[coll]] = 'MPI.p2p.'..coll end
      if MPI.async.p2p[coll] then fun2string[MPI.async.p2p[coll]] = 'MPI.async.p2p.'..coll end
   end
   for _, c in ipairs(cpuSel and {cpuSel} or {'cpu', 'gpu'}) do
      for _, s in ipairs(nodeSel and {nodeSel} or {'singlenode', 'multinode'}) do
         for _, a in ipairs(asyncSel and {asyncSel} or {'sync', 'async'}) do
            for _, coll in ipairs(collSel and {collSel} or {'allreduceTensor', 'broadcastTensor', 'reduceTensor', 'sendreceiveTensor'}) do
               local base = ('%s.%s.%s.%s'):format(c, s, a, coll)
               str = str .. ('MPI.collectiveSelector.%-40s \t->\t %s\n'):format(
                  base, fun2string[MPI.collectiveSelector[c][s][a][coll]])
            end
         end
      end
   end
   return str
end

return MPI
