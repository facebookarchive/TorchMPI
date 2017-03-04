--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
--[[

An extension of a Sequential container that keeps parameters in blocks of
contiguous memory.

]]--

local debug = false

-- torch class:
local BlockSequential, parent = torch.class('nn.BlockSequential', 'nn.Sequential')

function BlockSequential:__init(partitions, ...)
   assert(type(partitions) == 'number' and partitions > 0)
   nn.Sequential.__init(self, ...)
   self.metadata = {}
   self.metadata.numPartitions = partitions
end

-- function that returns blocks of parameters:
BlockSequential._partition = function(self)
   -- make sure this function gets called only once:
   if self.metadata.parameterBlocks then
      error('You can only call this function once.')
   end

   -- get parameter sizes for all modules:
   local paramsizes = {}
   local totalsize = 0
   self:applyToModules(function(m)
      local sz = 0
      local p, g = m:parameters()
      if p then
         for _, p in ipairs(m:parameters()) do sz = sz + p:nElement() end
      end
      table.insert(paramsizes, sz)
      totalsize = totalsize + sz
   end)
   local sizepersplit = math.ceil(totalsize / self.metadata.numPartitions)

   -- create parameter blocks:
   local blocks = {}
   self.metadata.parameterBlocks, self.metadata.gradparameterBlocks = {}, {}
   local blocksize, block = 0, nn.Sequential()
   local idx = 0
   self:applyToModules(function(m)
      idx = idx + 1
      -- if block full, call getParameters:
      if blocksize + paramsizes[idx] > sizepersplit then
         local p, g = block:parameters()
         if p then
            local theta, gradtheta = block:getParameters()
            table.insert(blocks, block)
            table.insert(self.metadata.parameterBlocks, theta)
            table.insert(self.metadata.gradparameterBlocks, gradtheta)
         end
         block = nn.Sequential()
         -- add module to new block:
         block:add(m)
         blocksize = paramsizes[idx]
      else
         -- add module to current block:
         block:add(m)
         blocksize = blocksize + paramsizes[idx]
      end
   end)
   -- Finish with remainder
   if block:size() > 0 then
      local p, g = block:parameters()
      if p then
         local theta, gradtheta = block:getParameters()
         table.insert(blocks, block)
         table.insert(self.metadata.parameterBlocks, theta)
         table.insert(self.metadata.gradparameterBlocks, gradtheta)
      end
   end
   self.modules = blocks
   self.gradInput = self.modules[1].gradInput
   self.output = self.modules[#self.modules].output
   if debug then print('Partitioned into', self) end
end

BlockSequential.forward = function(self, input)
   if not self.metadata.parameterBlocks then
      self:_partition()
   end
   return parent.forward(self, input)
end

BlockSequential.getParameterBlocks = function(self)
   self:_partition()
   return self.metadata.parameterBlocks, self.metadata.gradparameterBlocks
end

BlockSequential.parameterBlocks = function(self)
   if not self.metadata.parameterBlocks then
      self:_partition()
   end
   return self.metadata.parameterBlocks, self.metadata.gradparameterBlocks
end

BlockSequential.parameters = function(self)
   return self:parameterBlocks()
end

BlockSequential.backwardStep = function(self, input, gradOutput, scale)
   assert(torch.isTypeOf(self, 'nn.BlockSequential'), torch.type(self))
   assert(torch.isTensor(input))
   assert(torch.isTensor(gradOutput))
   assert(not scale or torch.type(scale) == 'number', torch.type(scale))
   if not self.metadata.parameterBlocks then
      error('BlockSequential must be partitioned before '..debug.getinfo(1, "n").name)
   end

   self.metadata.stepNumber = self.metadata.stepNumber or 0
   self.metadata.stepNumber = self.metadata.stepNumber + 1
   if self.metadata.stepNumber == 1 then
      self.metadata.currentGradOutput = gradOutput
   elseif self.metadata.stepNumber == #self.modules + 1 then
      -- Early exit for wrap around
      self.metadata.stepNumber = 0
      return nil, nil, nil
   end

   local moduleIndex = #self.modules - self.metadata.stepNumber + 1
   self.metadata.currentModule = self.modules[moduleIndex]
   self.metadata.previousModule = self.modules[moduleIndex - 1]

   local inputTensor = (self.metadata.stepNumber == #self.modules) and
      input or self.metadata.previousModule.output
   self.metadata.currentGradOutput =
      self:rethrowErrors(self.metadata.currentModule,
                         moduleIndex,
                         'backward',
                         inputTensor,
                         self.metadata.currentGradOutput,
                         scale)
   self.metadata.currentModule.gradInput = self.metadata.currentGradOutput

   return self.metadata.currentGradOutput,
      self.metadata.parameterBlocks[moduleIndex],
      self.metadata.gradparameterBlocks[moduleIndex]
end

-- function that implements zeroing of grad parameters:
BlockSequential.zeroGradParameters = function(self)
   if not self.metadata.parameterBlocks then
      error('BlockSequential must be partitioned before '..debug.getinfo(1, "n").name)
   end
   -- otherwise, zero parameter updates per block:
   for _,block in pairs(self.metadata.gradparameterBlocks) do block:fill(0) end
end

-- function that implements parameter updates:
BlockSequential.updateParameters = function(self, learningRate)
   if not self.metadata.parameterBlocks then
      error('BlockSequential must be partitioned before '..debug.getinfo(1, "n").name)
   end
   -- otherwise, do parameter updates per block:
   for idx, block in pairs(self.metadata.parameterBlocks) do
      block:add(-learningRate, self.metadata.gradparameterBlocks[idx])
   end
end

BlockSequential.__tostring__ = function(self)
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.BlockSequential ('..self.metadata.numPartitions..') '
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
