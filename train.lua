require 'torch'
require 'optim'
require 'xlua'
require 'image'
dofile 'etc.lua'

function crop_SR_LR_patches_vsdr2(imgPath, res)
  local r = math.random(2,4)
  local w_SRPatch = res 
  local h_SRPatch = res 

  local w_LRPatch = w_SRPatch/r
  local h_LRPatch = h_SRPatch/r

  local img = image.load(imgPath)
  local c = img:size(1)
  local h = img:size(2)
  local w = img:size(3)
  local Xmin = math.floor(torch.uniform(0,w - w_SRPatch))
  local Ymin = math.floor(torch.uniform(0,h - h_SRPatch))
  -- local Xmin = 0
  -- local Ymin = 0
  local Xmax = Xmin + w_SRPatch
  local Ymax = Ymin + h_SRPatch
  -- print(imgPath)
  --print(w, h, Xmin, Ymin, Xmax, Ymax)
  local SRPatch = image.crop(img, Xmin, Ymin, Xmax, Ymax)
  local LRPatch = image.scale(SRPatch, w_LRPatch, h_LRPatch)
  LRPatch = image.scale(LRPatch, w_SRPatch, h_SRPatch)
  --image.save("lr.jpg",LRPatch)
  --image.save("sr.jpg",SRPatch)
  SRPatch = SRPatch/255
  LRPatch = LRPatch/255
  input  = LRPatch
  target = SRPatch - LRPatch
  return input, target
end

function crop_SR_LR_patches_vsdr1(imgPath, res)
  local r = math.random(2,4)
  local w_SRPatch = res 
  local h_SRPatch = res 

  local w_LRPatch = w_SRPatch/r
  local h_LRPatch = h_SRPatch/r

  local img = image.load(imgPath)
  local c = img:size(1)
  local h = img:size(2)
  local w = img:size(3)
  local Xmin = math.floor(torch.uniform(0,w - w_SRPatch))
  local Ymin = math.floor(torch.uniform(0,h - h_SRPatch))
  -- local Xmin = 0
  -- local Ymin = 0
  local Xmax = Xmin + w_SRPatch
  local Ymax = Ymin + h_SRPatch
  -- print(imgPath)
  --print(w, h, Xmin, Ymin, Xmax, Ymax)
  local SRPatch = image.crop(img, Xmin, Ymin, Xmax, Ymax)
  local LRPatch = image.scale(SRPatch, w_LRPatch, h_LRPatch)
  LRPatch = image.scale(LRPatch, w_SRPatch, h_SRPatch)
  --image.save("lr.jpg",LRPatch)
  --image.save("sr.jpg",SRPatch)
  local input  = LRPatch/255
  local target = SRPatch/255
  return input, target
end

params, gradParams = model:getParameters()
optimState = {
    learningRate = lr,
    learningRateDecay = 0.0,
    weightDecay = wDecay,
    momentum = mmt,
}
optimMethod = optim.sgd
tot_error = 0
cnt_error = 0
epoch = 0

function train1(trainPath, trainSz, model_name)
    local time = sys.clock()
    
    tot_error = 0
    cnt_error = 0
    local iter_cnt = 0

    model:training()
    shuffle = torch.randperm(trainSz)
    
    local inputs = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)
    local targets = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)

    for t = 1,trainSz,batchSz do
        
        if t+batchSz-1 > trainSz then
            inputs = torch.CudaTensor(trainSz-t+1,inputDim,inputSz,inputSz)
            targets = torch.CudaTensor(trainSz-t+1,inputDim,inputSz,inputSz)
            curBatchDim = trainSz-t+1
        else
            curBatchDim = batchSz
        end
        --print ("batch size: #", curBatchDim)
        for i = t,math.min(t+batchSz-1,trainSz) do
            
            local input, target
            
            if model_name == "vsdr1" then
              input, target = crop_SR_LR_patches_vsdr1(trainPath[shuffle[j]], inputSz)
            end
            if model_name == "vsdr2" then
              input, target = crop_SR_LR_patches_vsdr2(trainPath[shuffle[j]], inputSz)
            end        
	          --print(trainPath[shuffle[i]])
	          --print(input:size())
	          --print(target:size())
            input = torch.reshape(input,inputDim,inputSz,inputSz)
            target = torch.reshape(target,inputDim,inputSz,inputSz)
                        
            inputs[i-t+1]:copy(input)
            targets[i-t+1]:copy(target)
        end        
        if epoch > 0 and epoch%20 == 0 then
            optimState.learningRate = optimState.learningRate * 0.1
        end
             
        local feval = function(x)
           if x ~= params then
              params:copy(x)
           end

           gradParams:zero()
           local output = model:forward(inputs)
           local err = criterion:forward(output,targets)
           model:backward(inputs,criterion:backward(output,targets))
           err = err/curBatchDim
           tot_error = tot_error + err
           cnt_error = cnt_error + 1
            
           gradParams:div(curBatchDim)
           gradParams:clamp(-lr_theta/optimState.learningRate,lr_theta/optimState.learningRate)
           return err,gradParams
        end

        optimMethod(feval, params, optimState)
        
        if iter_cnt % 100 == 0 then
            print("epoch: " .. epoch .. "/" .. epochNum .. " batch: " ..  t .. "/" .. trainSz .. " loss: " .. tot_error/cnt_error)
        end
        
        iter_cnt = iter_cnt + 1
    end
   
    if epoch % 2 == 0 then
        local filename = paths.concat(save_dir, modelName)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, model)
    end
end


function train2(trainPath, trainSz, model_name)
    local time = sys.clock()
    
    tot_error = 0
    cnt_error = 0
    local iter_cnt = 0

    model:training()
    shuffle = torch.randperm(trainSz)
    
    local inputs = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)
    local targets = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)
    curBatchDim = batchSz
    iterSz = 1000000
    for i = 1, iterSz do
        shuffle = torch.randperm(trainSz)
        for j = 1,batchSz do
            if model_name == "vsdr1" then
              input, target = crop_SR_LR_patches_vsdr1(trainPath[shuffle[j]], inputSz)
            end
            if model_name == "vsdr2" then
              input, target = crop_SR_LR_patches_vsdr2(trainPath[shuffle[j]], inputSz)
            end            
            input = torch.reshape(input,inputDim,inputSz,inputSz)
            target = torch.reshape(target,inputDim,inputSz,inputSz)
                        
            inputs[j]:copy(input)
            targets[j]:copy(target)
        end
        if epoch > 0 and epoch%20 == 0 then
            optimState.learningRate = optimState.learningRate * 0.1
        end
        
        local feval = function(x)
           if x ~= params then
              params:copy(x)
           end

           gradParams:zero()
           local output = model:forward(inputs)
           local err = criterion:forward(output,targets)
           model:backward(inputs,criterion:backward(output,targets))
           err = err/curBatchDim
           tot_error = tot_error + err
           cnt_error = cnt_error + 1
            
           gradParams:div(curBatchDim)
           gradParams:clamp(-lr_theta/optimState.learningRate,lr_theta/optimState.learningRate)
           return err,gradParams
        end 
        
        optimMethod(feval, params, optimState)
        
        if iter_cnt % 1000 == 0 then
            print("epoch: " .. epoch .. "/" .. epochNum .. " batch: " ..  i .. "/" .. iterSz .. " loss: " .. tot_error/cnt_error)
        end
        
        iter_cnt = iter_cnt + 1    
        
        if iter_cnt % 10000 == 0 then
            collectgarbage()
            local filename = paths.concat(save_dir, 'model_'..iter_cnt..".t7")
            os.execute('mkdir -p ' .. sys.dirname(filename))
            print('==> saving model to '..filename)
            torch.save(filename, model)
        end                 
    end
   

end


