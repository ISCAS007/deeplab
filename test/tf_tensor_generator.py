# -*- coding: utf-8 -*-
from src.dataset.dataset_pipeline import get_dataset_files,dataset_pipeline
from easydict import EasyDict as edict
import tensorflow as tf
from tqdm import trange,tqdm

if __name__ == '__main__':
    config=edict()
    config.num_threads=4
    config.batch_size=4
    config.edge_width=5
    
    img_files,label_files=get_dataset_files('cityscapes','train')
    
    img_files=img_files[0:20]
    label_files=label_files[0:20]
    dataset=dataset_pipeline(config,img_files,label_files,is_train=True)
    
    tf_dataset=tf.data.Dataset.from_generator(dataset.generator,
                                              output_types=(tf.float32,tf.float32,tf.float32,tf.string,tf.int32,tf.int32),
                                              output_shapes=(tf.TensorShape([None, None,3]), 
                                                             tf.TensorShape([1024, 2048]),
                                                             tf.TensorShape([1024, 2048]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([])))
    iterator=tf_dataset.make_one_shot_iterator()
    img,seg,edge,img_filename,height,width=iterator.get_next()
    print(img.shape,seg.shape,edge.shape)
    sess=tf.Session()
    for i in range(3):
        np_img,np_seg,np_edge,img_name=sess.run([img,seg,edge,img_filename])
        print(np_img.shape,np_seg.shape,np_edge.shape,img_name)
    
    img,seg,edge,img_filename,height,width=dataset.iterator()
    print(img.shape,seg.shape,edge.shape)
    for i in trange(len(dataset)+3):
        np_img,np_seg,np_edge,img_name,np_height,np_width=sess.run([img,seg,edge,img_filename,height,width])
        print(img_name,np_height,np_width)
        print(np_img.shape)
        
    