require 'torch'
require 'nn'
require 'cudnn'
require 'module/normalConv'
model = nn.Sequential()
-- msra init (mean = 0, std = sqrt(2/(input))) input = c * h * w
model:add(cudnn.normalConv(3,64,3,3,1,1,1,1,0,math.sqrt(2/(3*3*3))))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true)) 
for i = 1,18 do
  model:add(cudnn.normalConv(64,64,3,3,1,1,1,1,0,math.sqrt(2/(3*3*64))))
  model:add(nn.SpatialBatchNormalization(64))
  model:add(nn.ReLU(true))
end
model:add(cudnn.normalConv(64,3,3,3,1,1,1,1,0,math.sqrt(2/(3*3*64))))
model:add(nn.SpatialBatchNormalization(3))

criterion = nn.MSECriterion()
criterion.sizeAverage = false

--print(model)

cudnn.convert(model, cudnn)

model:cuda()
criterion:cuda()

cudnn.fastest = true
cudnn.benchmark = true

