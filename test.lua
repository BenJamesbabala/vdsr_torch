require 'torch'
require 'xlua' 
require 'optim'
require 'image'

local function MSE(x1, x2)
   return (x1 - x2):pow(2):mean()
end

local function PSNR(x1, x2)
   local mse = MSE(x1, x2)
   -- log10
   return 10 * (math.log((255*255) / mse)/math.log(10))
end
--[[
--]]
function gen_input_target(imgPath, scale)
  local img = image.load(imgPath)
  local h = img:size(2)
  local w = img:size(3)
  local h1 = h/scale
  local w1 = w/scale
  local img1 = image.scale(img, w1, h1)
  img1 = image.scale(img1, w, h)
  input  = img1/255
  target = img/255
  return input, target
end

function test(testPath, testSz, model_name)
    
    if mode == "test" then
        print("model loading...")
        model = torch.load(save_dir .. modelName)
    end
    
    model:evaluate()
    
    print('==> testing:')
    PSNR_sum = 0 
    for did = 1,testSz do
        
        local input,target
        input, target = gen_input_target(testPath[did],testScale)
        local sz = input:size()
        input = input:cuda()
        input = torch.reshape(input,1,inputDim,sz[2],sz[3])
        

        local output = model:forward(input)
        input = torch.reshape(input,inputDim,sz[2],sz[3])
        output = torch.reshape(output,outputDim,sz[2],sz[3])
        if model_name == "vsdr1" then
        
        end
        
        if model_name == "vsdr2" then
          output = output + input
        end
        
        
        input = input*255
        output = output*255

        target = target:cuda()
        target = torch.reshape(target,outputDim,sz[2],sz[3])
        target = target*255
        local cropSz = testScale
        output = image.crop(output:type('torch.FloatTensor'),cropSz,cropSz,sz[3]-cropSz,sz[2]-cropSz)
        target = image.crop(target:type('torch.FloatTensor'),cropSz,cropSz,sz[3]-cropSz,sz[2]-cropSz)
        input = image.crop(input:type('torch.FloatTensor'),cropSz,cropSz,sz[3]-cropSz,sz[2]-cropSz)
        

        image.save("result/input_" .. did .. ".jpg",input)
        image.save("result/output_" .. did .. ".jpg",output)
        image.save("result/target_" .. did .. ".jpg",target)
        
        PSNR_sum = PSNR_sum + PSNR(output,target)
    end

    print('PSNR: ' .. PSNR_sum/testSz)

end
