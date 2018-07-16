# spider_deep_sexer

Determine jumping spider sex with a ConvNet.

This is a simple cnn (convolutional neural network) approach to identify whether a spider in a top-down cropped photo is male or female. This was written with frame by frame video analysis in mind, where a fast and reliable way to id the sex of 'blobs' within the video can be helpful (for instance when a tracking program switches the identity of a subject after a close interaction). Currently working with Habronattus pyrrithrix at >98% accuracy, can be further pushed by retraining the model with a larger batch of training images.

The file hapy_sexer.h5 is an already pretrained neural network that can be used to classify an example image by running the file spider_sex_classify.R. The example image can be any aspect ratio or resolution, but should display a top-down view of a single H. pyrrithrix.
