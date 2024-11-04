## Abstract
此存储库展示了IPComposer 新的模型框架结构，旨在充分利用两种方法的优势，实现精确的图像生成以及丰富的文本和视觉条件。我们聚焦于两个分支，一个分支提供额外的交叉注意力层基于参考图像为图像生成提供全局的信息指导，另一个分支聚焦于优化的文本嵌入和实例图像的组合，提供局部的实例级别的信息指导，共同实现强化语义和强化图像指引的可控图像生成

## Usage
### <font style="color:rgb(31, 35, 40);">Environment Setup</font>
```python
conda create -n ipcomposer python
conda activate ipcomposer
pip install torch torchvision torchaudio
pip install transformers==4.25.1 accelerate datasets evaluate diffusers==0.16.1 xformers triton scipy clip gradio facenet-pytorch

python setup.py install
```

162上面的环境
```
conda activate fastcom
cd /home/capg_bind/96/mxf/workgroup/huawei-chanllenge/IPcomposer
bash scripts/run_training.sh
```

### <font style="color:rgb(31, 35, 40);">Download the Pre-trained Models</font>
总体保存的路径在

```python
96/mxf/workgroup/huawei-chanllenge/IPcomposer/outputs
```

LVIS_178 这个数据集包含 985 张图像，训练结果保存在

```python
96/mxf/workgroup/huawei-chanllenge/IPcomposer/outputs/lvis_178
```

LVIS_337 这个数据集包含 1462 张图像，训练结果保存在

```python
96/mxf/workgroup/huawei-chanllenge/IPcomposer/outputs/lvis_337
```

### Inference
```python
bash scripts/run_inference.sh
```

### <font style="color:rgb(31, 35, 40);">Training</font>
<font style="color:rgb(31, 35, 40);">Prepare the LVIS（fast composer-type dataset）training data:</font>

```python
FFHQ_DATAPATH=/home/capg_bind/96/mww/datasets/ffhq_wild_files
LVIS_178_DATAPATH=/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/rare_v3.0
LIVS_337_DATAPATH=/home/capg_bind/97/zfd/diffusion/ZFD_Huawei/rare_all_v1.0
```

<font style="color:rgb(31, 35, 40);">Run training:</font>

<font style="color:rgb(31, 35, 40);">单卡训练：</font>

```python
bash scripts/run_training_one_gpu.sh
```

<font style="color:rgb(31, 35, 40);">多卡训练</font>

```python
bash scripts/run_training.sh
```

## TODOs
- [ ] 337 数据集上面存在格式问题
- [ ] 1203 全类上面的训练
- [ ] 基于保存的 unet、ipadapter 权重去推理完成正确性验证
- [ ] 分阶段训练、联合训练训练策略的消融实验测评

