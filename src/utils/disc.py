# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf

def get_edge(ann_img, edge_width=5):
    kernel = np.ones((edge_width, edge_width), np.uint8)
    ann_edge = cv2.Canny(ann_img, 0, 1)
    ann_dilation = cv2.dilate(ann_edge, kernel, iterations=1)
    ann_dilation = (ann_dilation > 0).astype(np.uint8)
    return ann_dilation

def batch_get_edge(batch_ann_img,edge_width=5):
    b,h,w,c=batch_ann_img.shape
    assert c==1, 'the channel of ann image %d != 1'%c
    batch_edge=np.zeros_like(batch_ann_img)
    for idx in range(b):
        batch_edge[idx,:,:,0]=get_edge(batch_ann_img[idx,:,:,0].astype(np.uint8),edge_width)
    
    return batch_edge

def tf_batch_get_edge(session,tf_labels,name,edge_width=5):
    batch_ann_img=session.run(tf_labels)
    batch_edge=batch_get_edge(batch_ann_img,edge_width)
    return tf.convert_to_tensor(value=batch_edge,dtype=tf_labels.dtype,name=name)