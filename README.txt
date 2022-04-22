1. Extract_masks - generating masks with assigning values ​​to pixels in accordance with the class of objects (raster with 1 and 0 - for buildings and background, respectively, raster with 2 and 0 - for meadows and background, raster with 3 and 0 - for agricultural fields and background , raster with 4 and 0 for forests and background, raster with 5 and 0 for water surface and background).
2. Binary_masks - generation of a binary mask for each type of objects with pixel values ​​0 and 1 (0 - objects of this class are absent, 1 - objects of this class are present);
3. Multimask_creation - assigning zero to all pixels with a certain value or range of values;
4. Masked_rasters - calculation of the background corresponding to the background of the mask for satellite images (assigning the value 0 to the background);
5. Crop_image_gdal - raster cropping is not the required number of parts of arbitrary dimension;
6. Simple_multiclass_unet_model - neural network architecture - a function with prescribed layers of convolution, union and unfolding;
7. Multiclass_unet_sandstone - pre-processing of input data and launching the learning function prescribed in the previous program, as well as plotting graphs describing learning outcomes;
8. Testing - testing the model on a set of test images.