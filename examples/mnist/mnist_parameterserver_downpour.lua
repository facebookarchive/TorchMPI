--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the license found in the
 LICENSE-examples file in the root directory of this source tree.
--]]
require('nn')
require('paths')

local tnt = require('torchnet')

-- use GPU or not:
local cmd = torch.CmdLine()
cmd:option('-usegpu', false, 'use gpu for training')
cmd:option('-seed', 1111, 'use gpu for training')
cmd:option('-prefetch', 25, 'prefetch distance for asynchronous communications')
cmd:option('-tau', 50, 'communication cycle length for parameterserver (see easgd paper, we reuse the notation)')
cmd:option('-initDelay', 20, 'delay the first communication to let the networks search a bit independently')
cmd:option('-sendFrequency', 25, 'frequency at which we perform async sends')
cmd:option('-momentum', 0.9, 'see EASGD paper')

local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

local mpi = require('torchmpi')
-- The model we use for GPU + MPI is 1 Lua/Terra process for 1 GPU
-- mpi.start sets the GPU automatically
mpi.start(config.usegpu)
local mpinn = require('torchmpi.nn')
local parameterserver = require('torchmpi.parameterserver')

-- Set the random seed manually for reproducibility.
torch.manualSeed(config.seed)

-- set up logistic regressor:
local net = nn.Sequential():add(nn.Linear(784,10))
-- Perform weight and bias synchronization before starting training
mpinn.synchronizeParameters(net)
for _, v in pairs(net:parameters()) do mpi.checkWithAllreduce(v, 'initialParameters') end

local criterion = nn.CrossEntropyCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
local clerr  = tnt.ClassErrorMeter{topk = {1}}
engine.hooks.onStartEpoch = function(state)
   meter:reset()
   clerr:reset()
end
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print(string.format('[%d/%d] avg. loss: %2.4f; avg. error: %2.4f',
         mpi.rank() + 1, mpi.size(), meter:value(), clerr:value{k = 1}))
   end
end

local momentumTensorReferences = {}
engine.hooks.onBackward = function(state)
   if config.momentum and config.momentum ~= 0 then
      -- Disable Torchnet update rule once and for all
      state.lrSave = state.lrSave or state.lr
      state.lr = 0
   end
   local lr = state.lrSave or state.lr
   -- Create a local Downpour update rule if necessary
   state.downpourUpdate = state.downpourUpdate or
      mpi.DownpourUpdate{
         network = state.network,
         updateFrequency = config.tau,
         initDelay = config.initDelay,
         prefetch = config.prefetch,
         sendFrequency = config.sendFrequency,
         -- This applies locally just before sending to the server
         localUpdate = function(t) t:mul(-lr); return t end
      }
   -- Apply it
   state.downpourUpdate:update(state.t)
   -- Perform SGD step with momentum if necessary
   if config.momentum and config.momentum ~= 0 then
      local w, gw = state.network:parameters()
      for i = 1, #w do
         local p, g = w[i], gw[i]
         -- Nesterov's accelerated gradient rewritten as in Bengio's
         -- http://arxiv.org/pdf/1212.0901.pdf
         -- Note that originally momentumTensorReferences[p] = 0
         momentumTensorReferences[p] =
            momentumTensorReferences[p] or p:clone():zero()
         p:add(config.momentum * config.momentum, momentumTensorReferences[p])
            :add( -(1 + config.momentum) * lr, g)
         momentumTensorReferences[p]:mul(config.momentum):add(-lr, g)
      end
   end
end

-- set up GPU training:
if config.usegpu then
   cutorch.manualSeed(config.seed)

   -- copy model to GPU:
   require('cunn')
   net       = net:cuda()
   criterion = criterion:cuda()

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

local makeIterator = paths.dofile('makeiterator.lua')

-- train the model:
engine:train{
   network   = net,
   iterator  = makeIterator('train'),
   criterion = criterion,
   lr        = 0.2,
   maxepoch  = 5,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = makeIterator('test'),
   criterion = criterion,
}

-- Wait for all to finish before printing
mpi.barrier()

-- There is no real synchronization, checking allreduce does not makes sense
-- for _, v in pairs(net:parameters()) do mpi.checkWithAllreduce(v, 'final parameters') end

local loss = meter:value()
local err = clerr:value{k = 1}
print(string.format('[%d/%d] test loss: %2.4f; test error: %2.4f',
   mpi.rank() + 1, mpi.size(), loss, err))

-- There is no real synchronization, checking allreduce does not makes sense
-- mpi.checkWithAllreduce(loss, 'final loss')

mpi.stop()
