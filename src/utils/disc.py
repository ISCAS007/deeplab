# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf

def get_edge(ann_img, edge_width=5,edge_class_num=2,ignore_index=None):
    kernel = np.ones((edge_width, edge_width), np.uint8)
    ann_edge = cv2.Canny(ann_img, 0, 1)
    if ignore_index is not None:
        # remove ignore area in ann_img
        ann_edge[ann_img==ignore_index]=0
        
    ann_dilation = cv2.dilate(ann_edge, kernel, iterations=1)
    
    if edge_class_num==2:
        # fg=0, bg=1
        edge_label = (ann_dilation == 0).astype(np.uint8)
    else:
        # fg=0, bg=1,2,...,edge_class_num-1
        edge_label = np.zeros_like(ann_img)+edge_class_num-1
        for class_num in range(edge_class_num-1):
            edge_label[np.logical_and(ann_dilation>0,edge_label==(edge_class_num-1))]=class_num
            ann_dilation = cv2.dilate(ann_dilation, kernel, iterations=1)
    
    if ignore_index is not None:
        edge_label[ann_img==ignore_index]=ignore_index
        
    return edge_label

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