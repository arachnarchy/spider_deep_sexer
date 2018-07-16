library(keras)

## load trained model
model <- load_model_hdf5("hapy_sexer.h5")

## Classify a single image   ---------------------------------------------------

# load image and transform to 150x150
img_path <- "hapym.png" 

img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255

# display transformed image
plot(as.raster(img_tensor[1,,,]))

# Classify: 0 is female, 1 is male
model %>% predict(img_tensor)
