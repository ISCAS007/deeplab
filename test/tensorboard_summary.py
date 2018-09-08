# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from src.dataset.dataset_pipeline import get_dataset_files, dataset_pipeline
from easydict import EasyDict as edict
import tensorflow as tf
from tqdm import trange, tqdm


def variable_summaries(var,var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_%s'%var_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        tf.summary.image('image',tf.reshape(var,[1,1024,2048,1]))


if __name__ == '__main__':
    config = edict()
    config.num_threads = 4
    config.batch_size = 4
    config.edge_width = 5

    img_files, label_files = get_dataset_files('cityscapes', 'train')

    img_files = img_files[0:20]
    label_files = label_files[0:20]
    dataset = dataset_pipeline(config, img_files, label_files, is_train=True)

    ignore_label = 255
    num_classes = 19
    writer = tf.summary.FileWriter('/home/yzbx/tmp/logs/tensorflow/test002',graph=None)
    with tf.Graph().as_default() as graph:
        
        img, seg, edge, img_filename, height, width = dataset.iterator()
        print(img.shape, seg.shape, edge.shape)
        seg = tf.identity(seg, name='seg')
        predictions = seg
        predictions = tf.identity(predictions, name='predictions')

        global_step = tf.train.get_or_create_global_step()
        # Gather initial summaries.
#        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for miou,acc
        labels = graph.get_tensor_by_name('seg:0')
        predictions = graph.get_tensor_by_name('predictions:0')
#        predictions = tf.image.resize_bilinear(predictions,tf.shape(labels)[1:3],align_corners=True)

        labels = tf.reshape(labels, shape=[-1])
        predictions = tf.reshape(predictions, shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, ignore_label),name='weights')

        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights = 0.
        labels = tf.where(
            tf.equal(labels, ignore_label), tf.zeros_like(labels), labels)
        
        predictions = tf.where(
                tf.equal(predictions,ignore_label),tf.zeros_like(predictions), predictions)
        
        variable_summaries(labels,'labels')
        variable_summaries(predictions,'predictions')
        variable_summaries(weights,'weights')
        # Define the evaluation metric.
        metric_map = {}
        metric_map['miou'] = tf.metrics.mean_iou(
            predictions, labels, num_classes, weights=weights)
        metric_map['acc'] = tf.metrics.accuracy(
            labels=labels, predictions=predictions, weights=tf.reshape(weights, shape=[-1]))
        
        metrics = {}
        metrics['miou']=tf.identity(metric_map['miou'][0],name='hello')
        metrics['acc']=tf.identity(metric_map['acc'][0],name='world')
        for x in ['miou', 'acc']:
            tf.summary.scalar('metrics/%s' % x, metrics[x])

        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in trange(5):
            np_img, np_seg, np_edge, img_name, np_height, np_width, summary, metrics_np = sess.run(
                [img, seg, edge, img_filename, height, width, summary_op, metric_map])
            print(img_name, np_height, np_width)
            print(metrics_np['miou'][0],metrics_np['acc'][0])
#            print(metrics_np['miou'],metrics_np['acc'])
            writer.add_summary(summary, i)
        writer.add_graph(graph)
        writer.flush()
