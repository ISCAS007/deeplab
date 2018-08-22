timestr=`date "+%Y-%m-%d__%H-%M-%S"`
model_variant=xception_65
dataset=cityscapes
CURRENT_PATH=`pwd`
echo "current path is $CURRENT_PATH"
PATH_TO_INITIAL_CHECKPOINT=${CURRENT_PATH}/deeplab/datasets/weights/xception/model.ckpt
#PATH_TO_INITIAL_CHECKPOINT=/home/yzbx/tmp/logs/tensorflow/deeplab/cityscapes/xception_65/2018-08-16__15-30-42/model.ckpt
PATH_TO_TRAIN_DIR=${HOME}/tmp/logs/tensorflow/deeplab/${dataset}/${model_variant}/${timestr}
PATH_TO_DATASET=${CURRENT_PATH}/deeplab/datasets/cityscapes/tfrecord
python test/deeplab_test.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --fine_tune_batch_norm=False \
    --train_split="train" \
    --model_variant="${model_variant}" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=769 \
    --train_crop_size=769 \
    --train_batch_size=2 \
    --dataset="${dataset}" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET} \
    --dump=False
