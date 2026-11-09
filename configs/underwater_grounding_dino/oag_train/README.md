# OAG 分阶段训练（每阶段 12 epoch）

在仓库根目录、openmmlab 环境执行：

```bash
python tools/train.py configs/underwater_grounding_dino/oag_train/oag_double_resnet50_rgb_utdac2020_12e.py
python tools/train.py configs/underwater_grounding_dino/oag_train/oag_double_resnet50_sonar_uatd_coco_12e.py
python tools/train.py configs/underwater_grounding_dino/oag_train/oag_double_resnet50_fusion_rgbs50_12e.py
```

按顺序完成每次训练后再执行下一条。默认 checkpoint 链：

- RGB 输出 `work_dirs/oag_train/rgb_utdac2020/epoch_12.pth`。
- Sonar 的 `load_from` 指向 RGB 输出，保留 RGB 检测参数，更新声呐和共享文本参数。
- Fusion 的 `load_from` 指向 `work_dirs/oag_train/sonar_uatd_coco/epoch_12.pth`。

更换数据集时，修改各配置的 `data_root`、`metainfo`、标注路径、图像目录、`work_dir`，以及下一阶段的 `load_from`。切换阶段不要加 `--resume`；同一阶段中断恢复才使用它。命令行 `--work-dir` 会改变保存位置，需要同步修改下一阶段路径。

RGB 训练 backbone/neck、BERT/投影、RGB–文本 Encoder、候选生成、常规 Decoder 和检测头。Sonar 训练独立 backbone/投影、FFT 后声呐特征、逐类别热力图、BERT/投影；损失为 BCE+Dice 和仓库已有 RoIContrastiveFocalLoss，区域来自声呐 GT。未使用整图相同类别列表的 batch InfoNCE。Sonar 不执行 RGB Encoder/Decoder；默认关闭检测 AP 验证，保留每 epoch checkpoint。

所有阶段使用 OAGStaged，完整保留 OAGRGBPrompt 参数名称。DN label embedding 容量统一为 10（通过 bbox_head.num_classes），实际识别类别由数据集文本决定。更换为超过 10 类的数据时，需要统一提高三个阶段的容量后重新建立训练链。Fusion 配置继承原 RGBS50 数据和融合损失，恢复声呐排序权重 0.5、Decoder 提示权重 0.1、两个全局对比权重 0.25。

如使用原 `oag_double_resnet50_rgbs50_rgb_prompt_contrastive.py` 而不用本目录 Fusion 配置，必须同时把 `model.bbox_head.num_classes` 设为 10，并设置 `load_from` 指向 Sonar checkpoint，以匹配 DN embedding。推荐直接使用本目录 Fusion 配置。

`_base_staged.py` 统一 12 epoch、500 iteration warmup、8/11 epoch 衰减。两个 backbone 设置 frozen_stages=-1；BN 使用原 norm_eval。为便于阶段冻结及 DDP，关闭 encoder checkpoint（num_cp=0），相较原配置会增加显存占用，可先减小 batch_size。

验证：CPU 缩小模型完成 RGB loss/backward/predict、Sonar loss/backward、非活动 backbone 无梯度，以及三阶段 state_dict 严格互载。未执行完整数据集训练或 GPU/DDP 验证。
