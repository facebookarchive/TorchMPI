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

local config = cmd:parse(arg)
print(string.format('running on %s', config.usegpu and 'GPU' or 'CPU'))

local mpi = require('torchmpi')
-- The model we use for GPU + MPI is 1 Lua/Terra process for 1 GPU
-- mpi.start sets the GPU automatically
mpi.start(config.usegpu)

-- Set the random seed manually for reproducibility.
torch.manualSeed(config.seed)

-- Model-Parallel Linear layer
local MPLinear, parent = torch.class('nn.MPLinear', 'nn.Linear')
function MPLinear.__init(self, i, o)
   assert(i % mpi.size() == 0, ('i=%d not divisible by %d'):format(i, mpi.size()))
   nn.Linear.__init(self, i / mpi.size(), o)
end

local function narrowInput(input)
   local dim = input:nDimension()
   assert(input:size(dim) % mpi.size() == 0)
   local size = input:size(dim) / mpi.size()
   return input:narrow(dim, mpi.rank() * size + 1, size)
end

function MPLinear.updateOutput(self, input)
   local input = narrowInput(input)
   self.output = nn.Linear.updateOutput(self, input)
   mpi.allreduceTensor(self.output)
   return self.output
end

function MPLinear.updateGradInput(self, input, gradOutput)
   local input = narrowInput(input)
   self.gradInput = nn.Linear.updateGradInput(self, input, gradOutput)
   mpi.allreduceTensor(self.gradInput)
   return self.gradInput
end

function MPLinear.accGradParameters(self, input, gradOutput, scale)
   local input = narrowInput(input)
   nn.Linear.accGradParameters(self, input, gradOutput, scale)
end

-- set up logistic regressor:
local net = nn.Sequential():add(nn.MPLinear(784, 10, mpi.size()))
local criterion = nn.CrossEntropyCriterion()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
local clerr  = tnt.ClassErrorMeter{topk = {1}}
engine.hooks.onStartEpoch = function(state)
   meter:reset()
   clerr:reset()
end

local correctnessCheck = true
engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      print(string.format('[%d/%d] avg. loss: %2.4f; avg. error: %2.4f',
         mpi.rank() + 1, mpi.size(), meter:value(), clerr:value{k = 1}))
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
   iterator  = makeIterator('train', 'sequential'),
   criterion = criterion,
   lr        = 0.2,
   maxepoch  = 5,
}

-- measure test loss and error:
meter:reset()
clerr:reset()
engine:test{
   network   = net,
   iterator  = makeIterator('test', 'sequential'),
   criterion = criterion,
}

local loss = meter:value()
local err = clerr:value{k = 1}
print(string.format('[%d/%d] test loss: %2.4f; test error: %2.4f',
   mpi.rank() + 1, mpi.size(), loss, err))

mpi.stop()
