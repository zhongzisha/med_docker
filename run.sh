#!/bin/bash

INPUT_DIR=${1}
PROJ_NAME=${2}
DATA_ROOT=${INPUT_DIR}/.${PROJ_NAME}
SVS_ROOT=${DATA_ROOT}/svs
CSV_FILENAME=${DATA_ROOT}/all.xlsx

#rm -rf ${DATA_ROOT}
#if [ ! -d ${SVS_ROOT} ]; then
#  mkdir -p ${SVS_ROOT}
#fi
if [ ! -d ${DATA_ROOT} ]; then
  mkdir -p ${DATA_ROOT}
fi

if [ ! -e ${CSV_FILENAME} ]; then
  python3 get_data.py ${INPUT_DIR} ${DATA_ROOT}
fi

python3 create_patches.py \
  --source ${SVS_ROOT} \
  --save_dir ${DATA_ROOT} \
  --patch_size 256 \
  --preset tcga.csv \
  --seg --patch --stitch

IMAGE_EXT=".svs"
python3 extract_features.py \
  --data_h5_dir ${DATA_ROOT} \
  --data_slide_dir ${SVS_ROOT} \
  --csv_path ${CSV_FILENAME} \
  --feat_dir ${DATA_ROOT}/feats \
  --batch_size 512 \
  --target_patch_size 256 \
  --slide_ext ${IMAGE_EXT} \
  --model_name "mobilenetv3"

IMAGE_EXT=".svs"
FEATS_DIR=${DATA_ROOT}/feats/pt_files
ENCODER_CKPT_PATH="None"
NUM_CLUSTERS=10

if [ -e /app/web/data/${PROJ_NAME} ]; then
  rm -rf /app/web/data/${PROJ_NAME}
fi
ln -sf $DATA_ROOT /app/web/data/${PROJ_NAME}
python3 gen_projects.py

for NUM_PATCHES in 64; do
  for PATCH_MTL_SAVEPREFIX in "BRCAOnly2GPUsHF" "PanCancer2GPUsFP"; do #  "PanCancer2GPUsFP"
    SAVE_ROOT=${DATA_ROOT}/differential_results/$PATCH_MTL_SAVEPREFIX
    if [ $PATCH_MTL_SAVEPREFIX == "BRCAOnly2GPUsHF" ]; then
      BACKBONES=("mobilenetv3")
      EPOCH_NUMS=(54)
      SPLIT_NUM=1
      CLUSTER_TASK_INDEX=0
      CLUSTER_TASK_NAME="CLS_HistoAnno"
    else
      BACKBONES=("mobilenetv3")
      EPOCH_NUMS=(22)
      SPLIT_NUM=3
      CLUSTER_TASK_INDEX=0
      CLUSTER_TASK_NAME="TP53_cls"
    fi

    for bi in ${!BACKBONES[@]}; do
      BACKBONE=${BACKBONES[${bi}]}
      EPOCH_NUM=${EPOCH_NUMS[${bi}]}
      CKPT_PATH=snapshot_${EPOCH_NUM}.pt

      for ENCODER_TYPE in "imagenet"; do
        for CLUSTER_FEAT_NAME in "feat_before_attention_feat"; do #"feat_after_encoder_feat" "feat_before_attention_feat"
          for CLUSTER_STRATEGY in "density_log_3d"; do    #"density_log_2d" "density_log_3d" "feature_kmeans"

            python3 run_differential.py \
              --csv_filename ${CSV_FILENAME} \
              --dataset_root ${DATA_ROOT} \
              --image_size 224 \
              --save_root ${SAVE_ROOT} \
              --split_num "${SPLIT_NUM}" \
              --backbone ${BACKBONE} \
              --num_patches ${NUM_PATCHES} \
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

          done
        done
      done
    done

    cd $SAVE_ROOT
    bash /app/unzip.sh
    cd /app

  done
done


exit

docker build -t debug:1.0 .

docker run -d -it --name test --memory="32g" --memory-swap="64g" -v "/Users/zhongz2/data:/appdata" -p 8080:80 debug:1.0

docker exec -d -it test bash /app/run.sh /appdata/inputs test2_project

docker stop test && docker rm test && docker rmi debug:1.0


edit `/etc/docker/daemon.json`
```
{
        "data-root": "/mnt/disks/data1/docker_root/docker",
        "insecure-registries":["35.222.196.170:5000"]
}
```

sudo mkdir -p certs
openssl req -newkey rsa:4096 -nodes -sha256 -keyout certs/domain.key -addext "subjectAltName = IP:35.222.196.170" -x509 -days 365 -out certs/domain.crt
sudo docker run -d \
--restart=always \
--name registry \
-v "$(pwd)"/certs:/certs \
-e REGISTRY_HTTP_ADDR=0.0.0.0:443 \
-e REGISTRY_HTTP_TLS_CERTIFICATE=/certs/domain.crt \
-e REGISTRY_HTTP_TLS_KEY=/certs/domain.key \
-p 443:443 -p 5000:5000 \
registry:2



docker image push 35.222.196.170:5000/debug:2.0
curl -X GET http://35.222.196.170:5000/v2/_catalog
curl -X GET http://35.222.196.170:5000/v2/debug/tags/list
