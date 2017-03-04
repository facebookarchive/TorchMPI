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
local cache = require('torchmpi.cache')
local parameterserver = require('torchmpi.parameterserver.env')

local DownpourUpdate, Update =
   torch.class("torchmpi.parameterserver.DownpourUpdate",
               "torchmpi.parameterserver.Update",
               MPI)

DownpourUpdate.__init = argcheck {
   { name = "self", type = "torchmpi.parameterserver.DownpourUpdate" },
   { name = "network", type = "nn.Module" },
   { name = "shardingCommunicator", type = "number", default = 0 },
   { name = "dataparallelCommunicator", type = "number", default = 0 },
   { name = "updateFrequency", type = "number", default = 10  },
   { name = "initDelay", type = "number", default = 100 },
   { name = "prefetch", type = "number", default = 0 },
   { name = "sendFrequency", type = "number", default = 1 },
   { name = "localUpdate", type = "function" },
   call = function(self, network, shardingCommunicator, dataparallelCommunicator,
         updateFrequency, initDelay, prefetch, sendFrequency, localUpdate)
      Update.__init(
         self, network, shardingCommunicator, dataparallelCommunicator,
         updateFrequency, initDelay, prefetch)
      self.sendFrequency = sendFrequency
      self.nextSend = self.initDelay + self.sendFrequency
      -- Downpour needs an extra copy of parameters to save the old values
      -- before integration
      self.tensorReferences = {}
      -- Downpour applies learning rate locally to the accumulated
      -- gradients before sending, the parameterserver just needs to add
      self.localUpdate = localUpdate
   end
}

DownpourUpdate.__send = function(self, step)
   local p, g = self.network:parameters()
   for i, gw in ipairs(g) do
      self.tensorReferences[i] = self.tensorReferences[i] and
         self.tensorReferences[i]:add(gw) or gw:clone()
   end
   if step == self.nextSend then
      self.handlesSend = parameterserver.sendTensors(
         p, self.tensorReferences, 'add', self.localUpdate)
      self.handlesSend = parameterserver.syncHandles(self.handlesSend)
      for i, gw in ipairs(g) do
         self.tensorReferences[i]:zero()
      end
      self.nextSend = self.nextSend + self.sendFrequency
   end
end

DownpourUpdate.__integrate = function(self, step)
   if step == self.nextIntegration then
      -- Make sure prefetches completed
      self.handlesPrefetch =
         parameterserver.syncHandles(self.handlesPrefetch)
      -- Copy parameters from server locally
      local p, g = self.network:parameters()
      parameterserver.integrateTensors(
         p, function(fetched, t) t:copy(fetched) end)
      self.nextIntegration = self.nextIntegration + self.updateFrequency
      return true
   end
   return false
end
