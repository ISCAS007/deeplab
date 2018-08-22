# -*- coding: utf-8 -*-
from deeplab import common
#from deeplab import model
import tensorflow as tf
from deeplab.utils import train_utils
from deeplab.core import feature_extractor
from src.dataset.dataset_pipeline import get_dataset_files, dataset_pipeline, batch_preprocess_image_and_label, preprocess_image_and_label
from torch.utils import data as td
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from tqdm import trange,tqdm
#from deeplab.datasets import segmentation_dataset
import numpy as np
import time
import six
import os

slim = tf.contrib.slim

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'

DATASETS_CLASS_NUM = {
    'cityscapes': 19,
    'pascal_voc_seg': 21,
    'ade20k': 151,
}

DATASETS_IGNORE_LABEL = {
    'cityscapes': 255,
    'pascal_voc_seg': 255,
    'ade20k': 0,
}


class deeplab_base():
    def __init__(self, flags):
        self.flags = flags
        self.graph = tf.Graph()
        self.writer = None
        
    def init_summary_writer(self):
        if self.writer is None:
            time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
            log_dir=os.path.join(os.path.expanduser('~/tmp/logs/tensorflow'),'deeplab_base',self.flags.dataset,'001',time_str)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            config_str = self.flags.flags_into_string().replace(
                '\n', '\n\n').replace('  ', '\t')
            self.writer.add_text(tag='config', text_string=config_str)

    def _build_model(self, images, labels, num_classes):
        """Builds a clone of DeepLab.

        Args:
          inputs_queue: A prefetch queue for images and labels.
          outputs_to_num_classes: A map from output type to the number of classes.
            For example, for the task of semantic segmentation with 21 semantic
            classes, we would have outputs_to_num_classes['semantic'] = 21.
          ignore_label: Ignore label.

        Returns:
          A map of maps from output_type (e.g., semantic prediction) to a
            dictionary of multi-scale logits names to logits. For each output_type,
            the dictionary has keys which correspond to the scales and values which
            correspond to the logits. For example, if `scales` equals [1.0, 1.5],
            then the keys would include 'merged_logits', 'logits_1.00' and
            'logits_1.50'.
        """
        FLAGS = self.flags
        outputs_to_num_classes = {
            common.OUTPUT_TYPE: num_classes
        }
        ignore_label = 255
        model_options = common.ModelOptions(
            outputs_to_num_classes=outputs_to_num_classes,
            crop_size=FLAGS.train_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)
        outputs_to_scales_to_logits = multi_scale_logits(
            images,
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid,
            weight_decay=FLAGS.weight_decay,
            is_training=True,
            fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

        # Add name to graph node so we can add to summary.
        output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
        output_type_dict[MERGED_LOGITS_SCOPE] = tf.identity(
            output_type_dict[MERGED_LOGITS_SCOPE],
            name=common.OUTPUT_TYPE)

        for output, num_classes in six.iteritems(outputs_to_num_classes):
            for scale, logits in six.iteritems(outputs_to_scales_to_logits[output]):
                print(output, scale, logits.shape)

        losses = dict()
        for output, num_classes in six.iteritems(outputs_to_num_classes):
            loss = train_utils.add_softmax_cross_entropy_loss_for_each_scale(
                outputs_to_scales_to_logits[output],
                labels,
                num_classes,
                ignore_label,
                loss_weight=1.0,
                upsample_logits=FLAGS.upsample_logits,
                scope=output)
            losses[output] = loss

        return outputs_to_scales_to_logits, losses

    def train(self):
        self.init_summary_writer()
        FLAGS = self.flags
        dataset_split = 'train'
        img_files, label_files = get_dataset_files(
            FLAGS.dataset, dataset_split)

        config = edict()
        config.num_threads = 4
        config.batch_size = FLAGS.train_batch_size
        config.edge_width = 5
        dataset = dataset_pipeline(config, img_files, label_files)
        data_loader = td.DataLoader(
            dataset=dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=8)
        
        with self.graph.as_default():
            # Build the optimizer based on the device specification.
            learning_rate = train_utils.get_model_learning_rate(
                FLAGS.learning_policy, FLAGS.base_learning_rate,
                FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
                FLAGS.training_number_of_steps, FLAGS.learning_power,
                FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
            optimizer = tf.train.MomentumOptimizer(
                learning_rate, FLAGS.momentum)
    
            num_classes = DATASETS_CLASS_NUM[FLAGS.dataset]
            ignore_label = DATASETS_IGNORE_LABEL[FLAGS.dataset]
            
    #        images_shape = (FLAGS.train_batch_size,
    #                       FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], 3)
    #        labels_shape = (FLAGS.train_batch_size,
    #                       FLAGS.train_crop_size[0], FLAGS.train_crop_size[1], 1)
    
            images_shape = (FLAGS.train_batch_size,
                            1024, 2048, 3)
            labels_shape = (FLAGS.train_batch_size,
                            1024, 2048, 1)
    
            print('image shape is', images_shape)
            print('label shape is', labels_shape)
            images_placeholder = [tf.placeholder(
                dtype=tf.float32, shape=images_shape[1:]) for idx in range(FLAGS.train_batch_size)]
            labels_placeholder = [tf.placeholder(
                dtype=tf.int32, shape=labels_shape[1:]) for idx in range(FLAGS.train_batch_size)]
            placeholders = []
            placeholders.extend(images_placeholder)
            placeholders.extend(labels_placeholder)
    
            images_preprocess = []
            labels_preprocess = []
            for ip, lp in zip(images_placeholder, labels_placeholder):
                ppi, ppl = preprocess_image_and_label(
                    ip, lp, FLAGS, ignore_label, is_training=True)
                images_preprocess.append(ppi)
                labels_preprocess.append(ppl)
    
            images = tf.stack(values=images_preprocess, axis=0, name=common.IMAGE)
            labels = tf.stack(values=labels_preprocess, axis=0, name=common.LABEL)
            assert len(images.shape) == 4
            assert len(labels.shape) == 4
            print('image shape is', images.shape)
            print('label shape is', labels.shape)
            outputs_to_scales_to_logits, losses = self._build_model(
                images, labels, num_classes)
            total_loss = tf.reduce_mean(losses[common.OUTPUT_TYPE])
            for key,value in six.iteritems(losses):
                print('losses key and value',key,type(value))
                
            #eval
            all_predictions = {}
            for output in sorted(outputs_to_scales_to_logits):
                print('key for outputs_to_scales_to_logits',output)
                scales_to_logits = outputs_to_scales_to_logits[output]
                logits = tf.image.resize_bilinear(
                    scales_to_logits[MERGED_LOGITS_SCOPE],
                    FLAGS.train_crop_size,
                    align_corners=True)
                all_predictions[output] = tf.argmax(logits, 3)
                
            predictions = all_predictions[common.OUTPUT_TYPE]
            print('predictions shape',predictions.shape)
            predictions = tf.reshape(predictions, shape=[-1])
            trues = tf.reshape(labels, shape=[-1])
            print('trues shape',trues.shape)
            weights = tf.to_float(tf.not_equal(labels, ignore_label))
        
            # Set ignore_label regions to label 0, because metrics.mean_iou requires
            # range of labels = [0, dataset.num_classes). Note the ignore_label regions
            # are not evaluated since the corresponding regions contain weights = 0.
            trues = tf.where(
                tf.equal(trues, ignore_label), tf.zeros_like(trues), trues)
            # Define the evaluation metric.
            metric_map = {}
            metric_map['miou'] = tf.metrics.mean_iou(
                predictions, trues, num_classes, weights=weights)
        
            metrics_to_values, metrics_to_updates = (
                tf.contrib.metrics.aggregate_metric_map(metric_map))
            
            train_op=optimizer.minimize(total_loss)
            global_init_op=tf.global_variables_initializer()
            local_init_op=tf.local_variables_initializer()
        
        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)

        sess = tf.Session(config=session_config,graph=self.graph)
        sess.run(global_init_op)
        sess.run(local_init_op)
        
        epoches = 1+FLAGS.training_number_of_steps//len(data_loader)
        print('epoches is', epoches)
        print('step is', len(data_loader))
#        summaries.add(tf.summary.scalar('learning_rate', learning_rate))
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        for epoch in trange(epoches,desc='epoches'):
            loss_list=[]
            miou_list=[]
            for i, (torch_images, torch_labels, torch_edges) in enumerate(tqdm(data_loader,desc='step')):

                np_images = np.split(
                    torch_images.numpy(), FLAGS.train_batch_size, axis=0)
                np_labels = np.split(
                    torch_labels.numpy(), FLAGS.train_batch_size, axis=0)
                np_images = [i[0, :, :, :] for i in np_images]
                np_labels = [np.expand_dims(i[0, :, :], axis=-1)
                             for i in np_labels]

                np_values = []
                np_values.extend(np_images)
                np_values.extend(np_labels)

                _,np_loss,np_metrics,np_map,np_predicts,net_input,net_label=sess.run(fetches=[train_op, total_loss, metrics_to_values, metric_map, all_predictions,images,labels], feed_dict={
                         i: d for i, d in zip(placeholders, np_values)})
#                <class 'NoneType'> <class 'numpy.float32'> <class 'dict'>
#                print(type(np_op),type(np_loss),type(np_metrics))
                print('loss=',np_loss,'total miou=',np_metrics['miou'],'current miou=',np_map['miou'][0])
#                print('predict label index is',np.unique(np_predicts[common.OUTPUT_TYPE]))
#                print('net input range in',np.min(net_input),np.max(net_input))
#                print('net label range in',np.unique(net_label))
                loss_list.append(np_loss)
                miou_list.append(np_map['miou'][0])
            self.writer.add_scalar('%s/loss' % dataset_split,
                              np.mean(loss_list), epoch)
            self.writer.add_scalar('%s/miou' % dataset_split,
                              np.mean(miou_list), epoch)

    def val(self):
        FLAGS = self.flags
        dataset_split = 'val'
        img_files, label_files = get_dataset_files(
            FLAGS.dataset_name, dataset_split)

        config = edict()
        config.num_threads = 4
        config.batch_size = FLAGS.train_batch_size
        config.edge_width = 5
        dataset = dataset_pipeline(config, img_files, label_files)
        data_loader = td.DataLoader(
            dataset=dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=8)
        print('step', len(data_loader))

    def dump(self):
        FLAGS = self.flags
        num_classes = DATASETS_CLASS_NUM[FLAGS.dataset]
        ignore_label = DATASETS_IGNORE_LABEL[FLAGS.dataset]
        images_shape = (FLAGS.train_batch_size,
                        1024, 2048, 3)
        labels_shape = (FLAGS.train_batch_size,
                        1024, 2048, 1)

        print('image shape is', images_shape)
        print('label shape is', labels_shape)
        images_placeholder = [tf.placeholder(
            dtype=tf.float32, shape=images_shape[1:]) for idx in range(FLAGS.train_batch_size)]
        labels_placeholder = [tf.placeholder(
            dtype=tf.int32, shape=labels_shape[1:]) for idx in range(FLAGS.train_batch_size)]
        placeholders = []
        placeholders.extend(images_placeholder)
        placeholders.extend(labels_placeholder)

        images_preprocess = []
        labels_preprocess = []
        for ip, lp in zip(images_placeholder, labels_placeholder):
            ppi, ppl = preprocess_image_and_label(
                ip, lp, FLAGS, ignore_label, is_training=True)
            images_preprocess.append(ppi)
            labels_preprocess.append(ppl)

        images = tf.stack(values=images_preprocess, axis=0, name=common.IMAGE)
        labels = tf.stack(values=labels_preprocess, axis=0, name=common.LABEL)
        assert len(images.shape) == 4
        assert len(labels.shape) == 4
        print('image shape is', images.shape)
        print('label shape is', labels.shape)
        outputs_to_scales_to_logits, losses = self._build_model(
            images, labels, num_classes)
        
        total_loss = 0
        for loss in losses.values():
            total_loss += loss

#        optimizer=tf.train.AdamOptimizer()
        learning_rate = train_utils.get_model_learning_rate(
        FLAGS.learning_policy, FLAGS.base_learning_rate,
        FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
        FLAGS.training_number_of_steps, FLAGS.learning_power,
        FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, FLAGS.momentum)
        print('momentum',FLAGS.momentum)
        train_step=optimizer.minimize(total_loss)
        
        #logits/semantic/biases/Momentum
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if i.name.find('logits')>=0:
                print(i)  # i.name if you want just a name
        
        print('uninited global variables'+'*'*50)
        uninited_var=tf.report_uninitialized_variables(tf.global_variables())
        print(uninited_var)
        
        print('uninited collection variables'+'*'*50)
        uninited_var=tf.report_uninitialized_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print(uninited_var)


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
    """Gets the scopes for extra layers.

    Args:
      last_layers_contain_logits_only: Boolean, True if only consider logits as
      the last layer (i.e., exclude ASPP module, decoder module and so on)

    Returns:
      A list of scopes for extra layers.
    """
    if last_layers_contain_logits_only:
        return [LOGITS_SCOPE_NAME]
    else:
        return [
            LOGITS_SCOPE_NAME,
            IMAGE_POOLING_SCOPE,
            ASPP_SCOPE,
            CONCAT_PROJECTION_SCOPE,
            DECODER_SCOPE,
        ]


def scale_dimension(dim, scale):
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)


def multi_scale_logits(
        images,
        model_options,
        image_pyramid,
        weight_decay=0.0001,
        is_training=False,
        fine_tune_batch_norm=False):

    # Setup default values.
    if not image_pyramid:
        image_pyramid = [1.0]
    crop_height = (
        model_options.crop_size[0]
        if model_options.crop_size else tf.shape(images)[1])
    crop_width = (
        model_options.crop_size[1]
        if model_options.crop_size else tf.shape(images)[2])

    # Compute the height, width for the output logits.
    logits_output_stride = (
        model_options.decoder_output_stride or model_options.output_stride)

    logits_height = scale_dimension(
        crop_height,
        max(1.0, max(image_pyramid)) / logits_output_stride)
    logits_width = scale_dimension(
        crop_width,
        max(1.0, max(image_pyramid)) / logits_output_stride)

    # Compute the logits for each scale in the image pyramid.
    outputs_to_scales_to_logits = {
        k: {}
        for k in model_options.outputs_to_num_classes
    }

    for image_scale in image_pyramid:
        if image_scale != 1.0:
            scaled_height = scale_dimension(crop_height, image_scale)
            scaled_width = scale_dimension(crop_width, image_scale)
            scaled_crop_size = [scaled_height, scaled_width]
            scaled_images = tf.image.resize_bilinear(
                images, scaled_crop_size, align_corners=True)
            if model_options.crop_size:
                scaled_images.set_shape(
                    [None, scaled_height, scaled_width, 3])
        else:
            scaled_crop_size = model_options.crop_size
            scaled_images = images

        updated_options = model_options._replace(
            crop_size=scaled_crop_size)
        outputs_to_logits = _get_logits(
            scaled_images,
            updated_options,
            weight_decay=weight_decay,
            reuse=tf.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

        # Resize the logits to have the same dimension before merging.
        for output in sorted(outputs_to_logits):
            outputs_to_logits[output] = tf.image.resize_bilinear(
                outputs_to_logits[output], [logits_height, logits_width],
                align_corners=True)

        # Return when only one input scale.
        if len(image_pyramid) == 1:
            for output in sorted(model_options.outputs_to_num_classes):
                outputs_to_scales_to_logits[output][
                    MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
            return outputs_to_scales_to_logits

        # Save logits to the output map.
        for output in sorted(model_options.outputs_to_num_classes):
            outputs_to_scales_to_logits[output][
                'logits_%.2f' % image_scale] = outputs_to_logits[output]

    # Merge the logits from all the multi-scale inputs.
    for output in sorted(model_options.outputs_to_num_classes):
        # Concatenate the multi-scale logits for each output type.
        all_logits = [
            tf.expand_dims(logits, axis=4)
            for logits in outputs_to_scales_to_logits[output].values()
        ]
        all_logits = tf.concat(all_logits, 4)
        merge_fn = (
            tf.reduce_max
            if model_options.merge_method == 'max' else tf.reduce_mean)
        outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(
            all_logits, axis=4)

    return outputs_to_scales_to_logits


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
    """Extracts features by the particular model_variant.

    Args:
      images: A tensor of size [batch, height, width, channels].
      model_options: A ModelOptions instance to configure models.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
      concat_logits: A tensor of size [batch, feature_height, feature_width,
        feature_channels], where feature_height/feature_width are determined by
        the images height/width and output_stride.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    """
    features, end_points = feature_extractor.extract_features(
        images,
        output_stride=model_options.output_stride,
        multi_grid=model_options.multi_grid,
        model_variant=model_options.model_variant,
        depth_multiplier=model_options.depth_multiplier,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    if not model_options.aspp_with_batch_norm:
        return features, end_points
    else:
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }

        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,
                reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                depth = 256
                branch_logits = []

                if model_options.add_image_level_feature:
                    if model_options.crop_size is not None:
                        image_pooling_crop_size = model_options.image_pooling_crop_size
                        # If image_pooling_crop_size is not specified, use crop_size.
                        if image_pooling_crop_size is None:
                            image_pooling_crop_size = model_options.crop_size
                        pool_height = scale_dimension(image_pooling_crop_size[0],
                                                      1. / model_options.output_stride)
                        pool_width = scale_dimension(image_pooling_crop_size[1],
                                                     1. / model_options.output_stride)
                        image_feature = slim.avg_pool2d(
                            features, [pool_height, pool_width], [1, 1], padding='VALID')
                        resize_height = scale_dimension(model_options.crop_size[0],
                                                        1. / model_options.output_stride)
                        resize_width = scale_dimension(model_options.crop_size[1],
                                                       1. / model_options.output_stride)
                    else:
                        # If crop_size is None, we simply do global pooling.
                        pool_height = tf.shape(features)[1]
                        pool_width = tf.shape(features)[2]
                        image_feature = tf.reduce_mean(features, axis=[1, 2])[:, tf.newaxis,
                                                                              tf.newaxis]
                        resize_height = pool_height
                        resize_width = pool_width
                    image_feature = slim.conv2d(
                        image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
                    image_feature = tf.image.resize_bilinear(
                        image_feature, [resize_height, resize_width], align_corners=True)
                    # Set shape for resize_height/resize_width if they are not Tensor.
                    if isinstance(resize_height, tf.Tensor):
                        resize_height = None
                    if isinstance(resize_width, tf.Tensor):
                        resize_width = None
                    image_feature.set_shape(
                        [None, resize_height, resize_width, depth])
                    branch_logits.append(image_feature)

                # Employ a 1x1 convolution.
                branch_logits.append(slim.conv2d(features, depth, 1,
                                                 scope=ASPP_SCOPE + str(0)))

                if model_options.atrous_rates:
                    # Employ 3x3 convolutions with different atrous rates.
                    for i, rate in enumerate(model_options.atrous_rates, 1):
                        scope = ASPP_SCOPE + str(i)
                        if model_options.aspp_with_separable_conv:
                            aspp_features = split_separable_conv2d(
                                features,
                                filters=depth,
                                rate=rate,
                                weight_decay=weight_decay,
                                scope=scope)
                        else:
                            aspp_features = slim.conv2d(
                                features, depth, 3, rate=rate, scope=scope)
                        branch_logits.append(aspp_features)

                # Merge branch logits.
                concat_logits = tf.concat(branch_logits, 3)
                concat_logits = slim.conv2d(
                    concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
                concat_logits = slim.dropout(
                    concat_logits,
                    keep_prob=0.9,
                    is_training=is_training,
                    scope=CONCAT_PROJECTION_SCOPE + '_dropout')

                return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):
    """Gets the logits by atrous/image spatial pyramid pooling.

    Args:
      images: A tensor of size [batch, height, width, channels].
      model_options: A ModelOptions instance to configure models.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
      outputs_to_logits: A map from output_type to logits.
    """
    features, end_points = extract_features(
        images,
        model_options,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    if model_options.decoder_output_stride is not None:
        if model_options.crop_size is None:
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]
        else:
            height, width = model_options.crop_size
        decoder_height = scale_dimension(height,
                                         1.0 / model_options.decoder_output_stride)
        decoder_width = scale_dimension(width,
                                        1.0 / model_options.decoder_output_stride)
        features = refine_by_decoder(
            features,
            end_points,
            decoder_height=decoder_height,
            decoder_width=decoder_width,
            decoder_use_separable_conv=model_options.decoder_use_separable_conv,
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

    outputs_to_logits = {}
    for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_logits[output] = get_branch_logits(
            features,
            model_options.outputs_to_num_classes[output],
            model_options.atrous_rates,
            aspp_with_batch_norm=model_options.aspp_with_batch_norm,
            kernel_size=model_options.logits_kernel_size,
            weight_decay=weight_decay,
            reuse=reuse,
            scope_suffix=output)

    return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
    """Adds the decoder to obtain sharper segmentation results.

    Args:
      features: A tensor of size [batch, features_height, features_width,
        features_channels].
      end_points: A dictionary from components of the network to the corresponding
        activation.
      decoder_height: The height of decoder feature maps.
      decoder_width: The width of decoder feature maps.
      decoder_use_separable_conv: Employ separable convolution for decoder or not.
      model_variant: Model variant for feature extraction.
      weight_decay: The weight decay for model variables.
      reuse: Reuse the model variables or not.
      is_training: Is training or not.
      fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

    Returns:
      Decoder output with size [batch, decoder_height, decoder_width,
        decoder_channels].
    """
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
            reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
                feature_list = feature_extractor.networks_to_feature_maps[
                    model_variant][feature_extractor.DECODER_END_POINTS]
                if feature_list is None:
                    tf.logging.info('Not found any decoder end points.')
                    return features
                else:
                    decoder_features = features
                    for i, name in enumerate(feature_list):
                        decoder_features_list = [decoder_features]

                        # MobileNet variants use different naming convention.
                        if 'mobilenet' in model_variant:
                            feature_name = name
                        else:
                            feature_name = '{}/{}'.format(
                                feature_extractor.name_scope[model_variant], name)
                        decoder_features_list.append(
                            slim.conv2d(
                                end_points[feature_name],
                                48,
                                1,
                                scope='feature_projection' + str(i)))
                        # Resize to decoder_height/decoder_width.
                        for j, feature in enumerate(decoder_features_list):
                            decoder_features_list[j] = tf.image.resize_bilinear(
                                feature, [decoder_height, decoder_width], align_corners=True)
                            h = (None if isinstance(decoder_height, tf.Tensor)
                                 else decoder_height)
                            w = (None if isinstance(decoder_width, tf.Tensor)
                                 else decoder_width)
                            decoder_features_list[j].set_shape(
                                [None, h, w, None])
                        decoder_depth = 256
                        if decoder_use_separable_conv:
                            decoder_features = split_separable_conv2d(
                                tf.concat(decoder_features_list, 3),
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=weight_decay,
                                scope='decoder_conv0')
                            decoder_features = split_separable_conv2d(
                                decoder_features,
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=weight_decay,
                                scope='decoder_conv1')
                        else:
                            num_convs = 2
                            decoder_features = slim.repeat(
                                tf.concat(decoder_features_list, 3),
                                num_convs,
                                slim.conv2d,
                                decoder_depth,
                                3,
                                scope='decoder_conv' + str(i))
                    return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
    """Gets the logits from each model's branch.

    The underlying model is branched out in the last layer when atrous
    spatial pyramid pooling is employed, and all branches are sum-merged
    to form the final logits.

    Args:
      features: A float tensor of shape [batch, height, width, channels].
      num_classes: Number of classes to predict.
      atrous_rates: A list of atrous convolution rates for last layer.
      aspp_with_batch_norm: Use batch normalization layers for ASPP.
      kernel_size: Kernel size for convolution.
      weight_decay: Weight decay for the model variables.
      reuse: Reuse model variables or not.
      scope_suffix: Scope suffix for the model variables.

    Returns:
      Merged logits with shape [batch, height, width, num_classes].

    Raises:
      ValueError: Upon invalid input kernel_size value.
    """
    # When using batch normalization with ASPP, ASPP has been applied before
    # in extract_features, and thus we simply apply 1x1 convolution here.
    if aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                             'using aspp_with_batch_norm. Gets %d.' % kernel_size)
        atrous_rates = [1]

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            reuse=reuse):
        with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
            branch_logits = []
            for i, rate in enumerate(atrous_rates):
                scope = scope_suffix
                if i:
                    scope += '_%d' % i

                branch_logits.append(
                    slim.conv2d(
                        features,
                        num_classes,
                        kernel_size=kernel_size,
                        rate=rate,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope=scope))

            return tf.add_n(branch_logits)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.

    This operation differs from `tf.layers.separable_conv2d` as this operation
    applies activation function between depthwise and pointwise conv2d.

    Args:
      inputs: Input tensor with shape [batch, height, width, channels].
      filters: Number of filters in the 1x1 pointwise convolution.
      kernel_size: A list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
      rate: Atrous convolution rate for the depthwise convolution.
      weight_decay: The weight decay to use for regularizing the model.
      depthwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for depthwise convolution.
      pointwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for pointwise convolution.
      scope: Optional scope for the operation.

    Returns:
      Computed features after split separable conv2d.
    """
    outputs = slim.separable_conv2d(
        inputs,
        None,
        kernel_size=kernel_size,
        depth_multiplier=1,
        rate=rate,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=depthwise_weights_initializer_stddev),
        weights_regularizer=None,
        scope=scope + '_depthwise')
    return slim.conv2d(
        outputs,
        filters,
        1,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=pointwise_weights_initializer_stddev),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        scope=scope + '_pointwise')