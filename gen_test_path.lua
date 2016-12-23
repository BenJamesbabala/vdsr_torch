require 'image'
function prepImgs(datasetPath)
	local imgPaths = {}
	local imgNum = 0
	for file in paths.iterfiles(datasetPath) do
		imgNum = imgNum + 1
		local imPath = paths.concat(datasetPath, file)
		imgPaths[imgNum] = imPath
	end
	return imgPaths, imgNum
end

local imgBatch = {}
local datasetPath = "/home/gavinpan/workspace/dataset/sr/Set51/"
imgBatch.imgPaths, imgBatch.imgNum = prepImgs(datasetPath)
print(#imgBatch.imgPaths)
print(imgBatch.imgNum)

torch.save('testPath.t7', imgBatch)
