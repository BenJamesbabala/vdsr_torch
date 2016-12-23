require 'torch'
require 'nn'
require 'cudnn'
require 'module/normalConv'
model = nn.Sequential()
concat = nn.ConcatTable()
concat:add(nn.Identity())
subModel = nn.Sequential()  
-- msra init (mean = 0, std = sqrt(2/(input))) input = c * h * w
subModel:add(cudnn.normalConv(3,64,3,3,1,1,1,1,0,math.sqrt(2/(3*3*3))))
subModel:add(nn.SpatialBatchNormalization(64))
subModel:add(nn.ReLU(true))

for i = 1,18 do
subModel:add(cudnn.normalConv(64,64,3,3,1,1,1,1,0,math.sqrt(2/(3*3*64))))
subModel:add(nn.SpatialBatchNormalization(64))
subModel:add(nn.ReLU(true))
end
subModel:add(cudnn.normalConv(64,3,3,3,1,1,1,1,0,math.sqrt(2/(3*3*64))))
subModel:add(nn.SpatialBatchNormalization(3))

concat:add(subModel)
model:add(concat)
model:add(nn.CAddTable(false))

criterion = nn.MSECriterion()
criterion.sizeAverage = false

--print(model)

cudnn.convert(model, cudnn)

model:cuda()
criterion:cuda()

cudnn.fastest = true
cudnn.benchmark = true
