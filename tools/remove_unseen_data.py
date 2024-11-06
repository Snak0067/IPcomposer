import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Specify the paths
root = "/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/lvis_1203_add_a_photo_of"
image_ids_file = f"{root}/image_ids_train.txt"
output_file = f"{root}/00000/filtered_image_ids_train.txt"
data_dir = f"{root}/00000"

# Get all image IDs from the file
with open(image_ids_file, "r") as f:
    image_ids = f.read().splitlines()

# Function to check file existence for each image ID
def check_files(image_id):
    jpg_file = os.path.join(data_dir, f"{image_id}.jpg")
    json_file = os.path.join(data_dir, f"{image_id}.json")
    npy_file = os.path.join(data_dir, f"{image_id}.npy")
    
    # Return the image ID if all files exist, otherwise None
    if os.path.exists(jpg_file) and os.path.exists(json_file) and os.path.exists(npy_file):
        return image_id
    else:
        print(f"Missing file(s) for image ID: {image_id}")
        return None

# Initialize a list to store valid IDs
valid_image_ids = []

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor() as executor:
    # Submit tasks and add progress bar
    futures = {executor.submit(check_files, image_id): image_id for image_id in image_ids}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Checking files"):
        result = future.result()
        if result:
            valid_image_ids.append(result)

# Write valid IDs to the output file
with open(output_file, "w") as f:
    for image_id in valid_image_ids:
        f.write(f"{image_id}\n")

print(f"Filtered image IDs saved to {output_file}")
