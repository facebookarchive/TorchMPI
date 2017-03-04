--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]

require("nn")
local MPI = require("torchmpi.env")
local argcheck = require 'argcheck'
local mpinn = require('torchmpi.nn')
local cache = require('torchmpi.cache')
local parameterserver = require('torchmpi.parameterserver.env')

local Update = torch.class("torchmpi.parameterserver.Update", MPI)

Update.__init = argcheck {
   { name = "self", type = "torchmpi.parameterserver.Update" },
   { name = "network", type = "nn.Module" },
   { name = "shardingCommunicator", type = "number", default = 0 },
   { name = "dataparallelCommunicator", type = "number", default = 0 },
   { name = "updateFrequency", type = "number", default = 10  },
   { name = "initDelay", type = "number", default = 100 },
   { name = "prefetch", type = "number", default = 0 },
   call = function(self, network, shardingCommunicator, dataparallelCommunicator,
                   updateFrequency, initDelay, prefetch)
      assert(prefetch >= 0 and prefetch <= updateFrequency,
             'Prefetch must be in [0, '..updateFrequency..']') -- prefetch needs to make sense ..
      self.network = network
      self.shardingCommunicator = shardingCommunicator
      self.dataparallelCommunicator = dataparallelCommunicator

      self.initDelay = initDelay
      self.updateFrequency = updateFrequency
      self.prefetch = prefetch

      self.initParameterServer = initDelay
      self.nextPrefetch = initDelay + updateFrequency + prefetch
      self.nextIntegration = initDelay + updateFrequency

      self.handlesSend = {}
      self.handlesPrefetch = {}
   end
}

-- When it is time, init the  parameter server
Update.__shard = function(self, step)
   if step == self.initParameterServer then
      MPI.C.torchmpi_set_communicator(self.shardingCommunicator)
      local p = self.network:parameters()
      parameterserver.initTensors(p)
   end
end

-- If it is time to prefetch, do it
Update.__fetch = function(self, step)
   if step == self.nextPrefetch then
      self.handlesSend = parameterserver.syncHandles(self.handlesSend)
      local p = self.network:parameters()
      self.handlesPrefetch = parameterserver.prefetchTensors(p)
      self.nextPrefetch = self.nextPrefetch + self.updateFrequency
   end
end

-- Synchronize prefetch and integrate locally
-- Returns true if integration happened this step
Update.__integrate = function(self, step)
   error('NYI: Needs to be overriden')
end

Update.__send = function(self, step)
   error('NYI: Needs to be overriden')
end

Update.update = function(self, step)
   -- 1. Sharding only occurs once after initial warmup delay
   self:__shard(step)

   -- 2. If we combine parameter server with dataparallel,
   -- only the root makes parameter server updates
   local needBroadcast = 0
   local fetchIntegrate = false
   MPI.C.torchmpi_set_communicator(self.dataparallelCommunicator)
   if self.shardingCommunicator == self.dataparallelCommunicator then
      fetchIntegrate = true
   else
      -- If there is a distinct sharding and dataparallel communicator,
      -- only rank 0 on the dataparallelCommunicator integrates
      if MPI.rank() == 0 then fetchIntegrate = true end
   end

   -- 3. Fetch / integrate / send
   MPI.C.torchmpi_set_communicator(self.shardingCommunicator)
   if fetchIntegrate then
      self:__fetch(step)
      needBroadcast = self:__integrate(step) and 1 or 0
   end

   self:__send(step)

   -- 4. If we combine parameter server with dataparallel,
   -- and we integrated fetches, we need to broadcast from root
   if self.shardingCommunicator ~= self.dataparallelCommunicator then
      MPI.C.torchmpi_set_communicator(self.dataparallelCommunicator)
      needBroadcast = MPI.allreduce_double(needBroadcast)
      if needBroadcast > 0 then
         mpinn.synchronizeParameters(self.network)
         if self.debug then mpinn.checkWithAllreduce(self.network) end
      end
   end

   MPI.C.torchmpi_set_communicator(0)
end
