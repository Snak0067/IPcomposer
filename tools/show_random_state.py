import torch
from pathlib import Path

random_states_path = Path("/home/capg_bind/96/mxf/workgroup/huawei-chanllenge/IPcomposer/outputs/LVIS_1203/ipcomposer-localize-lvis-1_5-1e-5/checkpoint-500/random_states_0.pkl")

if random_states_path.exists():
    random_states = torch.load(random_states_path, map_location="cpu")
    print("Keys in random_states_0.pkl:", random_states.keys())
else:
    print(f"随机状态文件未找到：{random_states_path}")