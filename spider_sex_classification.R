# This is a simple convnet approach to identify whether a spider in
# a top-down cropped photo is male or female. This is with frame by frame video
# analysis in mind, where a fast and reliable way to id the sex of "blobs" 
# within the video can be helpful (for instance when a tracking program switches 
# the identity of a subject after a close interaction). 
# Currently working with Habronattus pyrrithrix at >98% accuracy.

library(keras)
library(stringr)
library(dplyr)

# split images into training, validation, and test sets ------------------------

# images from both sexes in one folder, filenames like male.27.jpg

# first make folder structure
original_dataset_dir <- "hapy"
base_dir <- "hapy_small"
dir.create(base_dir)

train_dir <- file.path(base_dir, "train")
dir.create(train_dir)

validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)

test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_female_dir <- file.path(train_dir, "female")
dir.create(train_female_dir)

train_male_dir <- file.path(train_dir, "male")
dir.create(train_male_dir)

validation_female_dir <- file.path(validation_dir, "female")
dir.create(validation_female_dir)

validation_male_dir <- file.path(validation_dir, "male")
dir.create(validation_male_dir)

test_female_dir <- file.path(test_dir, "female")
dir.create(test_female_dir)

test_male_dir <- file.path(test_dir, "male")
dir.create(test_male_dir)

# randomly select files (64:16:20 split), copy files into new folders
images <- list.files(original_dataset_dir)
n.female <- sum(str_detect(images, "female"))
n.male <- sum(!str_detect(images, "female"))

idx.f.train <- sample(1:n.female, n.female * 0.64)
idx.f.val <- sample((1:n.female)[-idx.f.train], n.female * 0.16)
idx.f.test <- sample((1:n.female)[-c(idx.f.train, idx.f.val)], n.female * 0.2)

idx.m.train <- sample(1:n.male, n.male * 0.64)
idx.m.val <- sample((1:n.male)[-idx.f.train], n.male * 0.16)
idx.m.test <- sample((1:n.male)[-c(idx.f.train, idx.f.val)], n.male * 0.2)

fnames <- paste0("female.", idx.f.train, ".png")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_female_dir))

fnames <- paste0("female.", idx.f.val, ".png")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_female_dir))

fnames <- paste0("female.", idx.f.test, ".png")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_female_dir))

fnames <- paste0("male.", idx.m.train, ".png")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(train_male_dir))

fnames <- paste0("male.", idx.m.val, ".png")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(validation_male_dir))

fnames <- paste0("male.", idx.m.test, ".png")
file.copy(file.path(original_dataset_dir, fnames),
          file.path(test_male_dir))



## set up convolutional neural network with dropout layer ----
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)


## Data augmentation ----

# This step does random transforms on the training images and is crucial for the
# small-ish number of images available.
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  channel_shift_range = 25,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

## display transformed example image if desired
# fnames <- list.files(train_female_dir, full.names = TRUE)
# img_path <- fnames[[3]]
# 
# img <- image_load(img_path, target_size = c(150, 150))
# img_array <- image_to_array(img)
# img_array <- array_reshape(img_array, c(1, 150, 150, 3))
# 
# augmentation_generator <- flow_images_from_data(
#   img_array,
#   generator = datagen,
#   batch_size = 1
# )
# 
# op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
# 
# for (i in 1:4) {
#   batch <- generator_next(augmentation_generator)
#   plot(as.raster(batch[1,,,]))
# }
# par(op)

## Generators ------
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 8,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 8,
  class_mode = "binary"
)

## train extended model with augmented data ----
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 10,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 5
)

# save out model weights if desired
# model %>% save_model_hdf5("hapy_sexer.h5") 

## run on test data -----
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 8,
  class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)

## Classify a specific image --------------------------------------------------------

# load image and transform to 150x150
img_path <- "example.png"
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor / 255

plot(as.raster(img_tensor[1,,,]))

# Classify: 0 is female, 1 is male
model %>% predict(img_tensor)
