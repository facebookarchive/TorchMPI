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

local EASGDUpdate, Update =
   torch.class("torchmpi.parameterserver.EASGDUpdate",
               "torchmpi.parameterserver.Update",
               MPI)

EASGDUpdate.__init = argcheck {
   { name = "self", type = "torchmpi.parameterserver.EASGDUpdate" },
   { name = "network", type = "nn.Module" },
   { name = "shardingCommunicator", type = "number", default = 0 },
   { name = "dataparallelCommunicator", type = "number", default = 0 },
   { name = "beta", type = "number", default = 0.9 },
   { name = "updateFrequency", type = "number", default = 10  },
   { name = "initDelay", type = "number", default = 100 },
   { name = "prefetch", type = "number", default = 0 },
   call =
      function(self, network, shardingCommunicator, dataparallelCommunicator,
               beta, updateFrequency, initDelay, prefetch)
      Update.__init(
         self, network, shardingCommunicator, dataparallelCommunicator,
         updateFrequency, initDelay, prefetch)
      self.beta = beta
      -- Send at each integration
      self.nextSend = self.nextIntegration
      -- EASGD needs an extra copy of parameters to save the old values
      -- before integration
      self.tensorReferences = {}
   end
}

-- No-op, we send immediately after integrating
EASGDUpdate.__send = function(self, step)
   local p, g = self.network:parameters()
   if step == self.nextSend then
      self.handlesSend = parameterserver.sendTensors(
         p, self.tensorReferences, 'add')
      -- self.handlesSend = parameterserver.syncHandles(self.handlesSend)
      self.nextSend = self.nextSend + self.updateFrequency
   end
end

-- Synchronize prefetch and integrate locally
EASGDUpdate.__integrate = function(self, step)
   if step == self.nextIntegration then
      -- Make sure prefetches completed, you can also play with disabling this
      self.handlesPrefetch = parameterserver.syncHandles(self.handlesPrefetch)
      local p, g  = self.network:parameters()
      parameterserver.integrateTensors(
         p,
         function(pref, t)
            local alpha = self.beta / MPI.size()
            self.tensorReferences[t] = self.tensorReferences[t] or t:clone()
            local old = self.tensorReferences[t]
            --- Paper version
            ----- old:copy(t):add(-pref)
            ----- t:add(-alpha, old)
            ----- old:mul(alpha)
            --- But pref may be a pinned CPU tensor so do this instead:
            old:copy(pref):add(-t)
            t:add(alpha, old)
            old:mul(-alpha)
         end
      )
      self.nextIntegration = self.nextIntegration + self.updateFrequency
      return true
   end
   return false
end
