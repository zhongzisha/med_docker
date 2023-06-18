#!/bin/bash

DATA_ROOT=$(pwd)/data/temp2
SVS_ROOT=${DATA_ROOT}/svs

rm -rf ${DATA_ROOT}
if [ ! -d ${SVS_ROOT} ]; then
  mkdir -p ${SVS_ROOT}
fi

python3 get_data.py $(pwd)/data/inputs ${DATA_ROOT}

python create_patches.py \
  --source ${SVS_ROOT} \
  --save_dir ${DATA_ROOT} \
  --patch_size 256 \
  --preset tcga.csv \
  --seg --patch --stitch

IMAGE_EXT=".svs"
python3 extract_features.py \
  --data_h5_dir ${DATA_ROOT} \
  --data_slide_dir ${SVS_ROOT} \
  --csv_path ${DATA_ROOT}/all.xlsx \
  --feat_dir ${DATA_ROOT}/feats \
  --batch_size 512 \
  --target_patch_size 256 \
  --slide_ext ${IMAGE_EXT} \
  --model_name "mobilenetv3"


PATCH_MTL_SAVEPREFIX="BRCAOnly2GPUsHF"
SAVE_ROOT=${DATA_ROOT}/differential_results
FEATS_DIR=${DATA_ROOT}/feats/pt_files
ENCODER_TYPE="imagenet"
ENCODER_CKPT_PATH="None"
CLUSTER_FEAT_NAME="feat_before_attention_feat"
CLUSTER_STRATEGY="density_log_3d"
NUM_CLUSTERS=10

if [ ${PATCH_MTL_SAVEPREFIX} == "BRCAOnly2GPUsHF" ]; then
  CLUSTER_TASK_INDEX=0
  CLUSTER_TASK_NAME="CLS_HistoAnno"
  EPOCH_NUM=54
  SPLIT_NUM=1
else # PanCancerOnly2GPUsHF
  CLUSTER_TASK_INDEX=0
  CLUSTER_TASK_NAME="TP53_cls"
  EPOCH_NUM=22
  SPLIT_NUM=3
fi
CKPT_PATH=snapshot_${EPOCH_NUM}.pt

python3 run_differential.py \
  --csv_filename ${DATA_ROOT}/all.xlsx \
  --action "train" --batch_size 4 \
  --dataset_root ${DATA_ROOT} \
  --image_size 224 \
  --save_root ${SAVE_ROOT} \
  --split_num "${SPLIT_NUM}" \
  --backbone "mobilenetv3" \
  --num_patches 256 \
  --num_channels 3 \
  --norm_type "mean_std" \
  --feats_dir ${FEATS_DIR} \
  --accum_iter 1 \
  --cache_root "None" \
  --split_label_name "None" \
  --encoder_type ${ENCODER_TYPE} \
  --encoder_ckpt_path ${ENCODER_CKPT_PATH} \
  --ckpt_path ${CKPT_PATH} \
  --use_weighted_sampling "False" \
  --cluster_feat_name ${CLUSTER_FEAT_NAME} \
  --cluster_strategy ${CLUSTER_STRATEGY} \
  --num_clusters ${NUM_CLUSTERS} \
  --cluster_task_index ${CLUSTER_TASK_INDEX} \
  --cluster_task_name ${CLUSTER_TASK_NAME} \
  --image_ext ${IMAGE_EXT}

exit;

docker build -t debug:1.0 .

docker run -d -it --name test -v "/Users/zhongz2/data/:/app/data" -p 8080:80 debug:1.0

