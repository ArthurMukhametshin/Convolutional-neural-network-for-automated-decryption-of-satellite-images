import rasterio.mask
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import numpy as np
import whitebox
import rasterio

red_raster = 'C:/OSM/NeuralNetwork/CNN/Images/Raw/Red.tif'

shape_buildings = 'C:/OSM/NeuralNetwork/CNN/Shapefiles/Buildings.shp'
shape_meadow_and_grass = 'C:/OSM/NeuralNetwork/CNN/Shapefiles/Meadow_and_grass.shp'
shape_farmland = 'C:/OSM/NeuralNetwork/CNN/Shapefiles/Farmland.shp'
shape_forests = 'C:/OSM/NeuralNetwork/CNN/Shapefiles/Forest.shp'
shape_water = 'C:/OSM/NeuralNetwork/CNN/Shapefiles/Water.shp'
shapes = [shape_buildings, shape_meadow_and_grass, shape_farmland, shape_forests, shape_water]

# открываем и читаем растр и шейп-файл
raster_path = red_raster
with rasterio.open(raster_path, 'r') as src:
    raster_img = src.read()
    raster_meta = src.meta

binary_rasters = [] # создаем пустой список

# генерируем полигон
def polygon_generation(polygon, transform):
    poly_pts = []
    poly = cascaded_union(polygon)
    for k in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(k))  # конвертируем полигон в изображение

    new_poly = Polygon(poly_pts)  # генерируем полигональный объект
    return new_poly

for shapefile in shapes:

    shape = gpd.read_file(shapefile)

    # генерируем бинарную маску
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in shape.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = polygon_generation(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = polygon_generation(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp, out_shape=im_size)

    # сохраняем маскированное изображение
    if shapefile == shape_buildings:
        objectclass = 'Buildings'
    elif shapefile == shape_meadow_and_grass:
        objectclass = 'Meadow_and_grass'
    elif shapefile == shape_farmland:
        objectclass = 'Farmland'
    elif shapefile == shape_forests:
        objectclass = 'Forests'
    else:
        objectclass = 'Water'

    mask = mask.astype('uint16')

    save_path = 'C:/OSM/NeuralNetwork/CNN/Images/Masks/binary_rasters/{0}.tif'.format(objectclass)
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    with rasterio.open(save_path, 'w', **bin_mask_meta) as dst:
            dst.write(mask * 1, 1)
    binary_rasters.append(save_path)