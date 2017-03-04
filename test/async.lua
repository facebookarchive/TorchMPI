--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')
require('nn')

local cmd = torch.CmdLine()
cmd:option('-gpu', false, 'run on gpu')
local config = cmd:parse(arg)
print(string.format('running on %s', config.gpu and 'GPU' or 'CPU'))

if config.gpu then
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.manualSeed(1)

local function dryRun(net, t, w, wSave, gw)
   -- Dry run to init tensors
   local o = net:forward(t)
   net:backward(t, o)
   w:copy(wSave)
   gw:zero()
   print('Done dryRun, got', t:mean(), w:mean(), gw:mean(), t:var(), w:var(), gw:var())
end

local mpi = require('torchmpi')
mpi.start(config.gpu)
local mpinn = require('torchmpi.nn')
if config.gpu then cutorch.manualSeed(1) end

local nInput = 100
local net = nn.Sequential()
   :add(nn.Linear(nInput, 200))
   :add(nn.Linear(200, 200))
   :add(nn.Linear(200, 400))
   :add(nn.Linear(400, 600))
   :add(nn.Linear(600, 800))
   :add(nn.Linear(800, 1000))
net:reset()
if config.gpu then net = net:cuda() end

local w, gw = net:getParameters()
local wSave = w:clone()
w:copy(wSave)





local t = config.gpu and
   torch.CudaTensor(96, nInput):normal() or
   torch.DoubleTensor(96, nInput):normal()

local nIter = 10
-- Dry run to init tensors
dryRun(net, t, w, wSave, gw)

if mpi.rank() == 0 then
   local timer = torch.Timer()
   timer:stop()
   timer:resume()
   for i = 1, nIter do
      o = net:forward(t)
      net:backward(t, o)
   end
   timer:stop()
   print('Sequential time: ', timer:time().real * 1000, ' gw: ', gw:mean(), gw:var())
end

mpi.barrier()






local timer = torch.Timer()
timer:stop()
local commtimer = torch.Timer()
commtimer:stop()

assert(96 % mpi.size() == 0)
local chunkSize = t:size(1) / mpi.size()
local tt = t:narrow(1, mpi.rank() * chunkSize + 1, chunkSize)
-- Dry run to init tensors
dryRun(net, tt, w, wSave, gw)

mpi.barrier()
timer:resume()
for i = 1, nIter do
   local o = net:forward(tt)
   net:backward(tt, o)
   commtimer:resume()
   mpinn.synchronizeGradients(net)
   commtimer:stop()
end
timer:stop()
mpi.barrier()

for i = 1, mpi.size() do
   if i == mpi.rank() + 1 then
      print('Parallel time: ', timer:time().real * 1000, ' of which comm: ', commtimer:time().real * 1000, ' gw: ', gw:mean(), gw:var())
   end
end





local timer = torch.Timer()
timer:stop()
local commtimer = torch.Timer()
commtimer:stop()

assert(96 % mpi.size() == 0)
local chunkSize = t:size(1) / mpi.size()
local tt = t:narrow(1, mpi.rank() * chunkSize + 1, chunkSize)
-- Dry run to init tensors
dryRun(net, tt, w, wSave, gw)

if config.gpu then
   mpi.async.initStreams(true)
end
mpinn.async.registerAsyncMPIBackward(net)

mpi.barrier()
for i = 1, nIter do
   timer:resume()
   local o = net:forward(tt)
   net:backward(tt, o)
   timer:stop()
   commtimer:resume()
   mpinn.async.synchronizeGradients(net)
   commtimer:stop()
end
timer:stop()
mpi.barrier()

for i = 1, mpi.size() do
   if i == mpi.rank() + 1 then
      print('Async parallel time: ', timer:time().real * 1000, ' of which comm: ', commtimer:time().real * 1000, ' gw: ', gw:mean(), gw:var())
   end
end


mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
