# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
from src.utils.disc import get_edge
import argparse
import os
import glob
import numpy as np
from torch.utils import data as td
from easydict import EasyDict as edict
from deeplab import input_preprocess
from deeplab import common

def get_dataset_files(dataset_name,dataset_split):
    if dataset_name == 'cityscapes':
        root='deeplab/datasets/cityscapes'
        img_files=glob.glob(os.path.join(root,'leftImg8bit',dataset_split,'**','*leftImg8bit.png'),recursive=True)
        label_files=glob.glob(os.path.join(root,'gtFine',dataset_split,'**','*labelTrainIds.png'),recursive=True)
        img_files.sort()
        label_files.sort()
    elif dataset_name == 'pascal_voc_seg':
        root='deeplab/datasets/pascal_voc_seg'
        assert False
    else:
        assert False
        
    print('dataset %s contain image %d, label %d'%(dataset_name,len(img_files),len(label_files)))
    assert len(img_files)==len(label_files),'number of image file is not equal to label file'
    
    return img_files,label_files

def preprocess_image_and_label(tf_image,tf_label,FLAGS,ignore_label,is_training=True):
    
    crop_size=FLAGS.train_crop_size
    original_image, image, label = input_preprocess.preprocess_image_and_label(
        tf_image,
        tf_label,
        crop_height=crop_size[0],
        crop_width=crop_size[1],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        min_scale_factor=FLAGS.min_scale_factor,
        max_scale_factor=FLAGS.max_scale_factor,
        scale_factor_step_size=FLAGS.scale_factor_step_size,
        ignore_label=ignore_label,
        is_training=is_training,
        model_variant=FLAGS.model_variant)
    
    return image,label

def batch_preprocess_image_and_label(numpy_image_4d,numpy_label_3d,FLAGS,ignore_label,is_training=True):
    b,h,w,c=numpy_image_4d.shape
    tf_images=[]
    tf_labels=[]
    for idx in range(b):
        tf_image=tf.convert_to_tensor(numpy_image_4d[idx,:,:,:],dtype=tf.float32)
        numpy_label_4d=np.expand_dims(numpy_label_3d,axis=-1)
        tf_label=tf.convert_to_tensor(numpy_label_4d[idx,:,:,:],dtype=tf.int32)
        
        pre_tf_image,pre_tf_label=preprocess_image_and_label(tf_image,tf_label,FLAGS,ignore_label,is_training)
        tf_images.append(pre_tf_image)
        tf_labels.append(pre_tf_label)
    
    return tf.stack(tf_images,name=common.IMAGE),tf.stack(tf_labels,name=common.LABEL)
        
class dataset_pipeline():
    def __init__(self,config,image_files,label_files):
        self.config=config
        self.num_threads=config.num_threads
        self.batch_size=config.batch_size
        self.edge_width=config.edge_width
        
        self.image_files=image_files
        self.label_files=label_files
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_file=self.image_files[index]
        label_file=self.label_files[index]
        
        img=cv2.imread(img_file,cv2.IMREAD_COLOR)
        label=cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)
        edge=get_edge(label,self.edge_width)
        
        return img,label,edge
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        help="dataset name",
                        choices=['cityscapes','voc'],
                        default='cityscapes')
    
    parser.add_argument("--dataset_split",
                        help="dataset split",
                        choices=['train','val','test'],
                        default='train')
    
    args=parser.parse_args()
    
    config=edict()
    config.num_threads=4
    config.batch_size=4
    config.edge_width=5
    
    img_files,label_files=get_dataset_files(args.dataset_name,args.dataset_split)
    
    dataset=dataset_pipeline(config,img_files,label_files)
    train_loader = td.DataLoader(
        dataset=dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=8)
    
    for i, (images, labels, edges) in enumerate(train_loader):
        print(i,images.shape,labels.shape,edges.shape)
        print(i,type(images),type(labels),type(edges))
        tf_images_4d=tf.convert_to_tensor(images.numpy(),tf.float32)
        tf_labels_3d=tf.convert_to_tensor(labels.numpy(),tf.int32)
        tf_labels_4d=tf.expand_dims(tf_labels_3d,axis=-1)
        tf_edges_3d=tf.convert_to_tensor(edges.numpy(),tf.int32)
        tf_edges_4d=tf.expand_dims(tf_edges_3d,axis=-1)
        
        print(i,tf_images_4d.shape,tf_labels_4d.shape,tf_edges_4d.shape)
        break