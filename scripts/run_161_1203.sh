export WANDB_NAME=ipcomposer-localize-lvis-1_5-1e-5
export WANDB_DISABLE_SERVICE=true
CUDA_VISIBLE_DEVICES=2,3,4,5,6

FFHQ_DATAPATH=/home/mxf/96_public/mww/datasets/ffhq_wild_files
LVIS_178_DATAPATH=/home/mxf/97/zfd/diffusion/ZFD_Huawei/rare_v3.0
LIVS_337_DATAPATH=/home/mxf/97/zfd/diffusion/ZFD_Huawei/lvis_all_v2.0
LVIS_1203_DATAPATH=/home/mxf/97/zfd/diffusion/ZFD_Huawei/lvis_1203_add_a_photo_of_check

DATASET_PATH=${LVIS_1203_DATAPATH}
DATASET_NAME="LVIS_1203_161"

# DATASET_PATH=${LVIS_178_DATAPATH}
# DATASET_NAME="lvis_178"

FAMILY=/home/mxf/96_public/zfd/0.hug/runwayml/
MODEL=stable-diffusion-v1-5
IMAGE_ENCODER=/home/mxf/96_public/zfd/0.hug/openai/clip-vit-large-patch14

accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 5 \
    --multi_gpu \
    train_ipcomposer.py \
    --pretrained_model_name_or_path ${FAMILY}/${MODEL} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir logs/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir outputs/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 50000 \
    --num_train_epochs 250 \
    --train_batch_size 3 \
    --learning_rate 1e-5 \
    --unet_lr_scale 1.0 \
    --checkpointing_steps 500 \
    --mixed_precision bf16 \
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
    --resume_from_checkpoint latest \
    --image_encoder_path ${IMAGE_ENCODER} \
    --train_ip_adapter \
    --report_to wandb # 本地调试先不用连通wandb
