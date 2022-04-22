import rasterio

def deletevalues(raster_in, raster_out):
    with rasterio.open(raster_in) as src:
        profile = src.profile
        raster = src.read()

    raster[raster > 1] = 0

    with rasterio.open(raster_out, 'w', **profile) as dst:
        dst.write(raster)

deletevalues('C:/OSM/NeuralNetwork/CNN/Images/Masks/binary_rasters/sum_1.tif', 'C:/OSM/NeuralNetwork/CNN/Images/Masks/binary_rasters/sum_2.tif')