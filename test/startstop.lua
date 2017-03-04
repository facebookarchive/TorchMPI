--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local cmd = torch.CmdLine()
cmd:option('-gpu', false, 'benchmark gpu')

local config = cmd:parse(arg)
local nRuns = config.check and 1 or 10

local mpi = require('torchmpi')
mpi.start(config.gpu)

print('Done start')
if mpi.rank() ~= 0 then -- 0 already prints by itself
   print('[MPI] Using the following hierarchical communicators:')
   print(mpi.communicatorNames())
end

mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
