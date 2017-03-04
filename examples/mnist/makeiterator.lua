--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the license found in the
 LICENSE-examples file in the root directory of this source tree.
--]]
local tnt = require('torchnet')

-- function that sets of dataset iterator:
local function getIterator(mode, sequential)
   -- load MNIST dataset:
   local mnist = require('mnist')
   local dataset = mnist[mode .. 'dataset']()
   dataset.data = dataset.data:reshape(dataset.data:size(1),
      dataset.data:size(2) * dataset.data:size(3)):double()
   local datasetToSplit =
      tnt.ListDataset{  -- replace this by your own dataset
         list = torch.range(1, dataset.data:size(1)):long(),
         load = function(idx)
            return {
               input  = dataset.data[idx],
               target = torch.LongTensor{dataset.label[idx] + 1},
            }  -- sample contains input and target
         end
      }

   -- Only split the dataset if we are in train mode
   -- Let everyone test on everything and assert outputs are the same
   local batchsize = 336 -- divisible by 8 and 9

   if sequential then
      return tnt.DatasetIterator{
         dataset = tnt.BatchDataset{ -- return batches of data:
            batchsize = batchsize,
            dataset = datasetToSplit
         }
      }
   end

   local mpi = require('torchmpi')
   if mode == 'train' and mpi.size() > 1 then
      local partitions = { }
      batchsize = mpi.size() and
         math.floor(batchsize / mpi.size()) or
         batchsize
      for i = 1, mpi.size() do
         partitions[tostring(i)] = 1.0 / mpi.size()
         print('PART size', 1.0 / mpi.size())
      end
      local initialpartition = tostring(mpi.rank() + 1) -- 0-based -> 1-based
      datasetToSplit = tnt.SplitDataset {
         partitions = partitions,
         initialpartition = initialpartition, -- 1-based
         dataset = datasetToSplit
      }
   end

   return tnt.DatasetIterator{
      dataset = tnt.BatchDataset{ -- return batches of data:
         batchsize = batchsize,
         dataset = datasetToSplit
      }
   }
end

return getIterator
