from osgeo import gdal

raster = gdal.Open('C:/OSM/NeuralNetwork/CNN/Images/Raw/New/NRG_masked.tif')
gt = raster.GetGeoTransform()

xmin = gt[0]
ymax = gt[3]
res = gt[1]
xlen = res * raster.RasterXSize
ylen = res * raster.RasterYSize

div = 42

xsize = xlen/div
ysize = ylen/div

xsteps = [xmin + xsize * i for i in range(div+1)]
ysteps = [ymax - ysize * i for i in range(div+1)]

a = 1
for i in range(div):
    for j in range(div):
        xmin = xsteps[i]
        xmax = xsteps[i+1]
        ymax = ysteps[j]
        ymin = ysteps[j+1]

        gdal.Warp(str(a) + '.tif', raster, outputBounds = (xmin, ymin, xmax, ymax))
        a += 1
