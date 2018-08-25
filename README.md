# dataset
```
export CITYSCAPES_DATASET=/xxx/datasets/cityscapes/gtFine_trainvaltest

datasets/cityscapes
├── cityscapesscripts -> /home/yzbx/git/gnu/cityscapesScripts/cityscapesscripts
├── gtFine -> gtFine_trainvaltest/gtFine
├── gtFine_trainvaltest
│   └── gtFine
├── leftImg8bit -> leftImg8bit_trainvaltest/leftImg8bit
├── leftImg8bit_trainvaltest
│   └── leftImg8bit
└── tfrecord

datasets/pascal_voc_seg
├── tfrecord
└── VOCdevkit
    └── VOC2012
    
```

# preprocess
- image
input to network with noraml range [0,255]
feature extractor will do preprocess when set preprocess_images=True

```
# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]

def _preprocess_subtract_imagenet_mean(inputs):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  return inputs - mean_rgb

def _preprocess_zero_mean_unit_range(inputs):
  """Map image values from [0, 255] to [-1, 1]."""
  return (2.0 / 255.0) * tf.to_float(inputs) - 1.0

_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
}

def extract_features(images,
                     output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     final_endpoint=None,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     num_classes=None,
                     global_pool=False):
  """Extracts features by the particular model_variant.

  Args:
    images: A tensor of size [batch, height, width, channels].
    output_stride: The ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    final_endpoint: The MobileNet endpoint to construct the network up to.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    preprocess_images: Performs preprocessing on images or not. Defaults to
      True. Set to False if preprocessing will be done by other functions. We
      supprot two types of preprocessing: (1) Mean pixel substraction and (2)
      Pixel values normalization to be [-1, 1].
    num_classes: Number of classes for image classification task. Defaults
      to None for dense prediction tasks.
    global_pool: Global pooling for image classification task. Defaults to
      False, since dense prediction tasks do not use this.

  Returns:
    features: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined
      by the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Unrecognized model variant.
  """
```

- label
use official code to create label image directly and then read it, no need to convert label if so. 
ignore_label=255

# change to official code
- [x] remove init_fn
- [x] load data from image file, not tfrecords
- [x] pipeline for image/edge/segmentation
- [ ] pipeline for train/val/test
- [ ] add edge support
    - [ ] ~ignore_label for edge~
- [ ] add global/branch support
- [ ] image_pyramid (default=None)
- [ ] upsample_logits (default=True)

# running demo
- ```sh test/train.sh```
- ```python src/run.py --logtostderr --training_number_of_steps=90000 --fine_tune_batch_norm=False --train_split=train --model_variant=xception_65 --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=769 --train_crop_size=769 --train_batch_size=2 --dataset=cityscapes```
