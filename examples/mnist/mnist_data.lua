--[[
 Copyright (c) 2016-present, Facebook, Inc.
 All rights reserved.

 This source code is licensed under the license found in the
 LICENSE-examples file in the root directory of this source tree.
--]]
local torch = require('torch')
require('paths')

local mnist = {}

local function readlush(filename)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()
   local ndim = f:readInt() - 0x800
   assert(ndim > 0)
   local dims = torch.LongTensor(ndim)
   for i=1,ndim do
      dims[i] = f:readInt()
      assert(dims[i] > 0)
   end
   local data = torch.ByteTensor(dims:storage())
   f:readByte(data:storage())
   f:close()
   return data
end

local function createdataset(dataname, labelname)
   local data = readlush(dataname)
   local label = readlush(labelname)
   assert(data:size(1) == label:size(1))
   local dataset = {data=data, label=label, size=data:size(1)}
   setmetatable(dataset, {__index=function(self, idx)
                                     assert(idx > 0 and idx <= self.size)
                                     return {x=data[idx], y=label[idx]}
                                  end})
   return dataset
end


function mnist.traindataset(path)
   return createdataset(paths.concat(path, 'data/train-images-idx3-ubyte'),
                        paths.concat(path, 'data/train-labels-idx1-ubyte'))
end

function mnist.testdataset(path)
   return createdataset(paths.concat(path, 'data/t10k-images-idx3-ubyte'),
                        paths.concat(path, 'data/t10k-labels-idx1-ubyte'))

end

return mnist
