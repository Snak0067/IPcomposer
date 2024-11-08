CAPTION="a dog <|image|> and a cat <|image|> walking on the lawn"
# A cat and a dog walking in the forest
DEMO_NAME="cat_dog"

CLIP_MODEL="/home/capg_bind/96/zfd/0.hug/openai/clip-vit-large-patch14"
# FASTCOMPOSER_MODEL=./outputs/lvis_178/ipcomposer-localize-lvis-1_5-1e-5/checkpoint-200
FASTCOMPOSER_MODEL=./outputs/LVIS_1203/ipcomposer-localize-lvis-1_5-1e-5/checkpoint-8000
# FASTCOMPOSER_MODEL=/home/capg_bind/96/mxf/workgroup/huawei-chanllenge/fastcomposer/model
# IPADAPTER_MODEL=/home/capg_bind/96/mxf/workgroup/huawei-chanllenge/IP-Adapter/trained_result_wo_number/checkpoint-100000/checkpoint-100000_ip_adapter.bin
IPADAPTER_MODEL=/home/capg_bind/96/mxf/workgroup/huawei-chanllenge/IPcomposer/outputs/LVIS_1203/ipcomposer-localize-lvis-1_5-1e-5/checkpoint-8000/checkpoint-8000_ip_adapter.bin
OUTPUT_NAME=lvis_1203_8000

# python inference_diffusers.py \
python inference.py \
    --pretrained_model_name_or_path /home/capg_bind/96/zfd/0.hug/runwayml/stable-diffusion-v1-5 \
    --finetuned_model_path ${FASTCOMPOSER_MODEL} \
    --test_reference_folder demo/test/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/inference/${DEMO_NAME} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path ${CLIP_MODEL}\
    --image_encoder_path ${CLIP_MODEL} \
    --num_image_tokens 1 \
    --max_num_objects 2 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --no_object_augmentation \
    --pretrained_ip_adapter_path ${IPADAPTER_MODEL} \
    --test_ip_adapter_image demo/test/cat_dog.jpg \
    --output_name ${OUTPUT_NAME}
