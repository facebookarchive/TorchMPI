--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local cmd = torch.CmdLine()
cmd:option('-gpu', false, 'run on gpu')
local config = cmd:parse(arg)

local mpi = require('torchmpi')
mpi.start(config.gpu)

if mpi.rank() == 0 then
   print('GPU available is:')
   print(mpi.collectiveSelectorToString('gpu'), "\n\n")

   print('GPU, single node, sync available is:')
   print(mpi.collectiveSelectorToString('gpu', 'singlenode', 'sync'), "\n\n")

   print('All available is:')
   print(mpi.collectiveSelectorToString())

   print('Success')
end
