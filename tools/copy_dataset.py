import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

source_dir = "/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/lvis_1203_add_a_photo_of_check"
dest_dir = "/data0/mxf/datasets/lvis"

# 确保目标目录存在
os.makedirs(dest_dir, exist_ok=True)

def copy_file(file_path):
    """拷贝单个文件到目标目录下保持相对路径结构"""
    dest_path = os.path.join(dest_dir, os.path.relpath(file_path, source_dir))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(file_path, dest_path)

def copy_files_in_parallel():
    # 获取所有文件路径
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # 使用多进程并带进度条进行文件拷贝
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(copy_file, all_files), total=len(all_files), desc="复制进度"))

if __name__ == "__main__":
    copy_files_in_parallel()
    print("文件复制完成。")
