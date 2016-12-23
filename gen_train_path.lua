require 'image'
-- img num in each file
num = 100 
function prepImageNet(ImageNetPath)
	local imgPaths = {}
	local imgNum = 0
	for dir in paths.iterdirs(ImageNetPath) do
		local c = 1
		for file in paths.iterfiles(paths.concat(ImageNetPath, dir)) do
			if c > num then 
			    break 
			end
			local imPath = paths.concat(ImageNetPath, dir, file)
			print(imPath)
			local img = image.load(imPath)
			if img:size(1) == 3 and img:size(2) > 288 and img:size(3) > 288 then  -- TODO global resolution 
				imgNum = imgNum + 1
				imgPaths[imgNum] = imPath
				c = c+1
				if imgNum % 100 == 0 then
				  print(imgNum)
				end
			end
		-- print(dir)
		end
	end
	return imgPaths, imgNum
end


local imgBatch = {}
local datasetPath = "/home/gavinpan/workspace/dataset/ILSVRC2012/sr_train/"
imgBatch.imgPaths, imgBatch.imgNum = prepImageNet(datasetPath)
print(#imgBatch.imgPaths)
print(imgBatch.imgNum)

torch.save('trainPath.t7', imgBatch)


