import whitebox
wbt = whitebox.WhiteboxTools()
wbt.verbose = True

nir = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/NIR.tif'
red = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/Red.tif'
green = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/Green.tif'
blue = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/Blue.tif'
mask = 'C:/OSM/NeuralNetwork/CNN/Images/Masks/binary_rasters/sum_2.tif'
masked_nir = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/NIR_masked.tif'
masked_red = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/Red_masked.tif'
masked_green = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/Green_masked.tif'
masked_blue = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/New/Blue_masked.tif'

rasters = [nir, red, green, blue]
masked_rasters = [masked_nir, masked_red, masked_green, masked_blue]

i = 0
for raster in rasters:
    wbt.multiply(raster, mask, masked_rasters[i], callback=None)
    i += 1