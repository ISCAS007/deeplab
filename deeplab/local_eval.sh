timestr=`date "+%Y-%m-%d__%H-%M-%S"`
model_variant=xception_65
dataset=cityscapes
CURRENT_PATH=`pwd`
echo "current path is $CURRENT_PATH"
PATH_TO_INITIAL_CHECKPOINT=${CURRENT_PATH}/deeplab/datasets/weights/xception/model.ckpt
#PATH_TO_INITIAL_CHECKPOINT=/home/yzbx/tmp/logs/tensorflow/deeplab/cityscapes/xception_65/2018-08-16__15-30-42/model.ckpt
PATH_TO_TRAIN_DIR=${HOME}/tmp/logs/tensorflow/deeplab/${dataset}/${model_variant}/2018-08-26__21-41-39
PATH_TO_EVAL_DIR=${HOME}/tmp/logs/tensorflow/deeplab/${dataset}/${model_variant}_eval/${timestr}
PATH_TO_DATASET=${CURRENT_PATH}/deeplab/datasets/cityscapes/tfrecord
# From tensorflow/models/research/
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --dataset="cityscapes" \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_logdir=${PATH_TO_EVAL_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
