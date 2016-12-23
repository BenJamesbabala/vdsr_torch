require 'torch'
require 'cunn'
dofile "etc.lua"
if model_name == "vsdr1" then
  dofile "model_vdsr1.lua"
end
if model_name == "vsdr2" then
  dofile "model_vdsr2.lua"
end

dofile "train.lua"
dofile "test.lua"  

local trainData
local trainLabel

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(0)

if mode == "train" then
    traindata = torch.load('trainPath.t7')
    testdata = torch.load('testPath.t7')
    fp_err = io.open("result/loss_" .. testScale .. ".txt","a")
    fp_PSNR = io.open("result/PSNR_" .. testScale .. ".txt","a")
    while epoch <= epochNum do
        --train1(traindata.imgPaths, traindata.imgNum)
        train2(traindata.imgPaths, traindata.imgNum, model_name)
        epoch = epoch + 1
        err = tot_error/cnt_error
        fp_err:write(err,"\n")
        test(testdata.imgPaths, testdata.imgNum, model_name)
        fp_PSNR:write(PSNR_sum/testdata.imgNum,"\n")
    end
    fp_err:close()
    fp_PSNR:close()
end


if mode == "test" then
    test()
end
