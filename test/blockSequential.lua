--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require 'nn'
local paths = require 'paths'
require('torchmpi.BlockSequential')

-- create network:
local N, D = 1000, 10
local numblocks = 7
local network = nn.BlockSequential(numblocks)
for n = 1,31 do network:add(nn.Linear(D, D)) end

-- perform forward-backward pass on network:
local input = torch.randn(N, D)
local gradoutput = torch.randn(N, D)
local output = network:forward(input):clone()
local gradinput = network:updateGradInput(input, gradoutput):clone()

-- create parameter blocks:
local p, dp = network:parameterBlocks()
-- may be a remainder block
assert((#p == numblocks or #p == numblocks + 1) and (#dp == numblocks or #dp == numblocks + 1))
for idx, block in pairs(p) do
   assert(block:isSameSizeAs(dp[idx]))
end

-- perform forward-backward pass on network:
local output2 = network:forward(input):clone()
assert(output:isSameSizeAs(output2))
assert(torch.add(output, -output2):abs():sum() < 1e-5)
local gradinput2 = network:updateGradInput(input, gradoutput):clone()
assert(gradinput:isSameSizeAs(gradinput2))
assert(torch.add(gradinput, -gradinput2):abs():sum() < 1e-5)

-- perform forward-backward pass on network:
local output3 = network:forward(input):clone()
assert(output:isSameSizeAs(output3))
assert(torch.add(output, -output3):abs():sum() < 1e-5)
local gradinput3 = nil
while gradoutput do
   gradinput3 = gradoutput
   gradoutput = network:backwardStep(input, gradoutput)
end
assert(gradinput:isSameSizeAs(gradinput3))
assert(torch.add(gradinput, -gradinput3):abs():sum() < 1e-5)

-- confirm that updateParameters works:
network:updateParameters(1)

print('Success')
