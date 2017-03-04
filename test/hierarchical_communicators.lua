--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
require('torch')

local cmd = torch.CmdLine()
cmd:option('-numRanks', 1, 'number or ranks')
cmd:option('-numNodes', 2, 'number or nodes')
local config = cmd:parse(arg)

local mpi = require('torchmpi')
local mpicache = require('torchmpi.cache')

-- This is an example of setting up a custom communicator say for performing
-- parameter server operations on top of synchronous SGD communicators.
-- A customCommunicatorInit function should return the result of
-- mpi.C.torchmpi_push_communicator with your preferred string for discriminating
-- amongst participants.
-- Processes with the same string will end up in the same (intra) group.
-- Processes with rank 0 within each group form another (inter) group.
--
-- Run like this:
-- mpirun -n 32 --map-by node --bind-to none --hostfile /etc/JARVICE/nodes luajit ./test/hierarchical_communicators.lua -numRanks 32 -numNodes 2

local div = 3
local function customCommunicatorInit()
   local res =
      mpi.C.torchmpi_push_communicator(tostring(mpi.rank() % div));
   assert(res == 1)
   return res
end

mpi.start{ withCuda = false, withIPCGroups = false, customCommunicatorInit = customCommunicatorInit, withCartesianCommunicator = true }
-- Creating a custom communicator will leave you at the level it has been created
-- Get back to level 0 to do stuff form the top.
mpi.C.torchmpi_set_communicator(0)
-- Level 0 is always cartesian
assert(mpi.C.torchmpi_is_cartesian_communicator())

assert(mpi.size() == config.numRanks,
       ("%d (expected %d)"):format(mpi.size(), config.numRanks))

local rankG = mpi.rank()

-- For the first level we check that we have the proper intracomm rank modulo div
mpi.C.torchmpi_set_communicator(1)
local rankL1 = mpi.rank()
local sizeL1 = mpi.size()
local iscartesian = config.numRanks < div or (config.numRanks % div == 0)
assert(iscartesian or not mpi.C.torchmpi_is_cartesian_communicator(),
   ("\nERROR on communicator\n%s%d mod %d = %d however iscartesian: %s"):format(
      mpi.communicatorNames(), config.numRanks, div, config.numRanks % div, mpi.C.torchmpi_is_cartesian_communicator())
)
assert(math.floor(rankG / div) == rankL1,
       ('\nERROR on communicator\n%sglobalRank: %d (div by %d gives %d) VS rankL1 %d'):format(
          mpi.communicatorNames(), rankG, div, math.floor(rankG / div), rankL1))

-- For the second level, we check that we have the proper intracomm rank modulo numNodes
mpi.C.torchmpi_set_communicator(2)
local rankL2 = mpi.rank()
local sizeL2 = mpi.size()
local iscartesian = (sizeL2 < config.numNodes) or (sizeL1 % config.numNodes == 0)
assert(iscartesian or not mpi.C.torchmpi_is_cartesian_communicator(),
   ("\nERROR on communicator\n%s%d mod %d = %d however iscartesian: %s"):format(
      mpi.communicatorNames(), sizeL1, config.numNodes, config.numRanks % config.numNodes, mpi.C.torchmpi_is_cartesian_communicator())
)
assert(math.floor(rankL1 / config.numNodes) == rankL2,
       ('\nERROR on communicator\n%srankL1: %d (div by %d gives %d) VS rankL2 %d'):format(
          mpi.communicatorNames(), rankL1, config.numNodes, math.floor(rankL1 / div), rankL2))

-- Now go back to level 0 to do other stuff
mpi.C.torchmpi_set_communicator(0)

mpi.barrier()
if mpi.rank() == 0 then print('Success') end
mpi.stop()
