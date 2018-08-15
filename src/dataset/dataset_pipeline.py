# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
from src.utils.disc import get_edge

class dataset_pipeline():
    def __init__(self,config,img_files,label_files):
        self.config=config
        self.num_threads=config.num_threads
        self.batch_size=config.batch_size
        self.edge_width=config.edge_width
        
        dataset = tf.data.Dataset.from_tensor_slices((img_files, label_files))
        dataset = dataset.shuffle(len(img_files))
        dataset = dataset.map(self.parse_function, num_parallel_calls=self.num_threads)
#        dataset = dataset.map(train_preprocess, num_parallel_calls=4)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        
    def parse_function(self,img_file, label_file):
        img=cv2.imread(img_file,cv2.IMREAD_COLOR)
        label=cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)
        edge=get_edge(label,self.edge_width)
        return img,label,edge