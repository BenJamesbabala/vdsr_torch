mode = "train"
model_name = "vsdr1"
--model_name = "vsdr2"
modelName = "model.net"

save_dir = "output"

trainScale = {2,3,4}
testScale = 4

inputSz = 41
inputDim = 3
outputDim = 3
fDim = 64
n = 18
lr_theta = 2e-3


lr = 1e-1
wDecay = 1e-4
mmt = 9e-1
--64
batchSz = 64
--80
epochNum = 1

