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
local nRuns = 100

local mpi = require('torchmpi')
local parameterserver = require('torchmpi.parameterserver')
mpi.start(config.gpu)

assert(not config.gpu, 'ParameterServer + GPU not supported atm')

for i = 1, nRuns do

   --------------------------------------------------------------------------------------------
   -- 1. Test initialization and default values.
   -- Each process launched by mpirun has its own mpi.rank().
   -- Collectively there are mpi.size() such tensors, one per process.
   -- We can synchronize them through the parameter server.
   local size = 1024 -- one test without edge cases 2^10
   local t = torch.FloatTensor(size):fill(mpi.rank())

   -- This is a collective, it performs a barrier
   local ps = parameterserver.init(t)
   t:fill(0)
   parameterserver.syncHandle(parameterserver.receive(ps, t))
   -- This barrier should be unnecessary (because only local parameterserver.syncHandle is needed)
   -- mpi.barrier()

   -- In particular, for this case, it means the default initialization for parameter
   -- server is:
   --    shard[0]   shard[1]    ...  shard[mpi.size()]
   --       0          1                mpi.size()
   assert(t:min() == 0, t:min())
   assert(t:max() == mpi.size() - 1, t:max())

   -- This is a collective, it performs a barrier
   parameterserver.free(ps)

   --------------------------------------------------------------------------------------------
   -- 2. Test > 1-D tensor, if the tensor is contiguous all is good
   local size1, size2 = 911, 101 -- large enough, prime enough to stress test boundaries
   local val = 123
   local t = torch.FloatTensor(size1, size2):fill(val)
   -- Different tensor size, need a new parameterserver!

   -- This is a collective, it performs a barrier
   local ps = parameterserver.init(t)
   t:zero()
   parameterserver.syncHandle(parameterserver.receive(ps, t))
   -- This barrier should be unnecessary (because only local parameterserver.syncHandle is needed)
   -- mpi.barrier()

   -- Every single process initialized its own tensor to `val`
   -- As a consequence the default initialization for the parameter server makes all shards
   -- be filled with val
   --    shard[0]   shard[1]    ...  shard[mpi.size()]
   --       val          val            val
   assert(t:min() == val, tostring(t:min()) .. ' VS ' .. tostring(val))
   assert(t:max() == val, tostring(t:max()) .. ' VS ' .. tostring(val))

   -- This is a collective, it performs a barrier
   parameterserver.free(ps)

   --------------------------------------------------------------------------------------------
   -- 3. Test send with zero update rule from a single process.
   -- Remember we are in distributed mode, mpi.size() processes run in parallel
   -- Now we need to start synchronizing things to perform more advanced tests
   local val = 123
   local t = torch.FloatTensor(size1, size2):fill(val)

   -- This is a collective, it performs a barrier
   local ps = parameterserver.init(t)

   -- Only the process with the last rank sends to the parameter server
   -- Otherwise this is a racy update and we can only say:
   --   0 <= val <= mpi.size() -1
   if mpi.rank() == mpi.size() - 1 then
      parameterserver.syncHandle(parameterserver.send(ps, t, 'zero'))
   end

   -- The combination of the syncHandle above and the mpi.barrier() call
   -- guarantees that everyone finished its update rule
   mpi.barrier()

   -- Now all processes receive and check
   parameterserver.syncHandle(parameterserver.receive(ps, t))
   -- This barrier should be unnecessary (because only local parameterserver.syncHandle is needed)
   -- mpi.barrier()

   assert(t:min() == 0, t:min() .. 'VS' .. tostring(val))
   assert(t:max() == 0, t:max() .. 'VS' .. tostring(val))

   -- This is a collective, it performs a barrier
   parameterserver.free(ps)

   --------------------------------------------------------------------------------------------
   -- 4. Test send with copy update rule
   -- Remember we are in distributed mode, mpi.size() processes run in parallel
   -- Now we need to start synchronizing things to perform more advanced tests
   local val = 123
   local t = torch.FloatTensor(size1, size2):fill(val)

   -- This is a collective, it performs a barrier
   local ps = parameterserver.init(t)
   t:fill(mpi.size() - 1)

   -- Only the process with the last rank sends to the parameter server
   -- Otherwise this is a racy update and we can only say:
   --   0 <= val <= mpi.size() -1
   if mpi.rank() == mpi.size() - 1 then
      parameterserver.syncHandle(parameterserver.send(ps, t, 'copy'))
   end

   -- The combination of the syncHandle above and the mpi.barrier() call
   -- guarantees that everyone finished its update rule
   mpi.barrier()

   -- Now all processes receive and check
   parameterserver.syncHandle(parameterserver.receive(ps, t))
   -- This barrier should be unnecessary (because only local parameterserver.syncHandle is needed)
   -- mpi.barrier()

   assert(t:min() == mpi.size() - 1, t:min() .. 'VS' .. tostring(val))
   assert(t:max() == mpi.size() - 1, t:max() .. 'VS' .. tostring(val))

   -- This is a collective, it performs a barrier
   parameterserver.free(ps)


   --------------------------------------------------------------------------------------------
   -- 5. Test copy + all add
   -- Remember we are in distributed mode, mpi.size() processes run on parallel
   -- Now we need to start synchronizing things to perform more advanced tests
   local val = 123
   local t = torch.FloatTensor(size1, size2):fill(val)

   -- This is a collective, it performs a barrier
   local ps = parameterserver.init(t)
   t:fill(mpi.rank())

   if mpi.rank() == mpi.size() - 1 then
      -- Only the process with the last rank sends to the parameter server
      parameterserver.syncHandle(parameterserver.send(ps, t, 'copy'))
   end

   -- The combination of the syncHandle above and the mpi.barrier() call
   -- guarantees that everyone finished its update rule
   mpi.barrier()

   -- All perform an add, the individual adds are unordered.
   -- Test assumes commutativity and associativity
   parameterserver.syncHandle(parameterserver.send(ps, t, 'add'))

   -- The combination of the syncHandle above and the mpi.barrier() call
   -- guarantees that everyone finished its update rule
   mpi.barrier()

   -- Put some garbage that must be overwritten
   t:fill(123)

   -- Now all receive and check
   parameterserver.syncHandle(parameterserver.receive(ps, t))
   -- This barrier should be unnecessary (because only local parameterserver.syncHandle is needed)
   -- mpi.barrier()

   local val = mpi.size() - 1 + ((mpi.size() - 1) * mpi.size()) / 2
   assert(t:min() == val, t:min() .. 'VS' .. tostring(val))
   assert(t:max() == val, t:max() .. 'VS' .. tostring(val))

   -- This is a collective, it performs a barrier
   parameterserver.free(ps)
end

if mpi.rank() == 0 then print('Success') end

-- Finally, terminate cleanly
mpi.stop()
