--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
--]]
local mpi = require('torchmpi')
local mpicache = require('torchmpi.cache')

local tester = {}

tester.runOneConfig = function(tests, nRuns, nSkip, config, printDebug)
   ---------- Test loop
   for testName, T in pairs(tests) do
      if not T.implemented then goto continueTestLoop end

      local lower = 8
      local upper = 23
      for i = lower, upper do -- (latency -> bw bound test)
         mpi.barrier()

         local pre = config.gpu and cutorch.getMemoryUsage() or nil
         local size = 2 ^ i + math.floor(math.random(128)) -- give it some kick
         local input = config.gpu and torch.CudaTensor(size) or torch.FloatTensor(size)

         input:fill(mpi.rank())
         local output = config.inPlace and input or input:clone():normal()

         local timer = torch.Timer()
         timer:stop()
         for r = 1, nRuns do
            if printDebug then
               print(
                  string.format(
                     '%s (run %d) (%s processes) (inPlace: %s GPU: %s Async: %s ) %s (%d contiguous values)',
                     testName,
                     r,
                     mpi.size(),
                     config.inPlace,
                     config.gpu,
                     config.async,
                     torch.type(input),
                     input:nElement()
                  )
               )
            end
            -- Skip the timing of the first nSkip iterations
            if r >= nSkip + 1 then timer:resume() end
            mpi.barrier()
            local inputClone = not config.inPlace and input:clone() or nil
            local handle = T.test(input, output, r == 1)
            mpi.barrier()
            if handle == 'NYI' then
               print('NYI: ', testName, size, 'skipping all such tests from now on')
               goto continueTestLoop
            end
            if handle == 'UNAVAILABLE' then
               print('UNAVAILABLE: ', testName, size, 'skpping all such tests from now on')
               goto continueTestLoop
            end
            if handle then
               mpi.syncHandle(handle)
            end
            timer:stop()

            if r == 1 then
               T.check(input, output, inputClone)
            end
         end

         if mpi.rank() == 0 and config.benchmark then
            local gpuMemString = config.gpu and
               string.format(
                  ' GPU memory pre / post / total alloc: %d / %d / %d',
                  pre,
                  cutorch.getMemoryUsage()) or
               ''
            local sizeFloat = 4
            local meanTimeS =  timer:time().real / (nRuns - nSkip)
            local commVolumeGB = T.communicationVolumeGB(input)
            local timingStr = string.format(
               '%s (%s processes) %s (%d contiguous values) -> %d us ',
               testName,
               mpi.size(),
               torch.type(input),
               input:nElement(),
               meanTimeS * 1e6)
            local throughputStr = string.format(
               '(%f GB/s assuming good implementation, i.e. %f MB through the slowest wire, assuming p2p wires)',
               commVolumeGB / meanTimeS,
               commVolumeGB * 1e3,
               timer:time().real / (nRuns - nSkip))
            print(timingStr .. throughputStr .. gpuMemString)
         end

         if printDebug then
            print("Barrier pre freeDescriptors")
         end
         mpi.barrier()
         if mpicache and config.gpu then mpicache.freeDescriptors() end
         mpi.barrier()
      end

      ::continueTestLoop::
   end
end

return tester
