export WANDB_NAME=ipcomposer-localize-lvis-1_5-1e-5
export WANDB_DISABLE_SERVICE=true
CUDA_VISIBLE_DEVICES=2,3,4,5

FFHQ_DATAPATH=/home/capg_bind/96/mww/datasets/ffhq_wild_files
LVIS_178_DATAPATH=/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/rare_v3.0
LIVS_337_DATAPATH=/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/lvis_all_v2.0
LVIS_1203_DATAPATH=/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/lvis_1203_add_a_photo_of

DATASET_PATH=${FFHQ_DATAPATH}
DATASET_NAME="ffhq"

# DATASET_PATH=${LVIS_178_DATAPATH}
# DATASET_NAME="lvis_178"

FAMILY=/home/capg_bind/96/zfd/0.hug/runwayml/
MODEL=stable-diffusion-v1-5
IMAGE_ENCODER=/home/capg_bind/96/zfd/0.hug/openai/clip-vit-large-patch14

accelerate launch \
    --mixed_precision=fp16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11435 \
    --num_processes 4 \
    --multi_gpu \
    train_ipcomposer.py \
    --pretrained_model_name_or_path ${FAMILY}/${MODEL} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir logs/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir outputs/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 20000 \
    --num_train_epochs 250 \
    --train_batch_size 4 \
    --learning_rate 1e-5 \
    --unet_lr_scale 1.0 \
    --checkpointing_steps 2000 \
    --mixed_precision fp16 \
    --allow_tf32 \
    --keep_only_last_checkpoint \
    --keep_interval 10000 \
    --seed 42 \
    --image_encoder_type clip \
    --image_encoder_name_or_path ${IMAGE_ENCODER} \
    --num_image_tokens 1 \
    --max_num_objects 4 \
    --train_resolution 512 \
    --object_resolution 224 \
    --text_image_linking postfuse \
    --object_appear_prob 0.9 \
    --uncondition_prob 0.1 \
    --object_background_processor random \
    --disable_flashattention \
    --train_image_encoder \
    --image_encoder_trainable_layers 2 \
    --mask_loss \
    --mask_loss_prob 0.5 \
    --object_localization \
    --object_localization_weight 1e-3 \
    --object_localization_loss balanced_l1 \
    --image_encoder_path ${IMAGE_ENCODER} \
    --train_ip_adapter \
    --report_to wandb \
    # --resume_from_checkpoint latest 
