# 对比损失消融

基准为 `oag_train/oag_double_resnet50_single_fusion_rgbs50_finetune_20e.py`。
两组只覆盖独立对比损失的开关/权重，不改模型类、Query Selector、FFT、模态分支及其余损失。
不修改任何 `mmdet` 源码。

## 实验设置

| 设置 | enable_legacy_contrastive | 声呐–文本权重 | 声呐–RGB 权重 |
| --- | --- | --- | --- |
| baseline（既有） | True | 0.25 | 0 |
| `01_no_contrastive` | False | 0 | 0 |
| `02_add_sonar_rgb` | True | 0.25 | 0.25 |

新增声呐–RGB 权重取 0.25，与 baseline 声呐–文本项同量级，是本次明确的默认实验选择，
不是原 baseline 已有值，也不是经过调参得出的最优值。可统一覆盖
`model.loss_sonar_image_weight` 做后续权重敏感性实验。

当前融合模型为 OAGStaged(training_stage='fusion') → OAGRGBPrompt。
实际全局对比项由 OAGRGBPrompt._compute_contrastive_losses 直接调用
SonarBindPrism._compute_contrastive_losses，不经过 OAG 中移除声呐–RGB loss key 的实现；
因此只用配置就能打开声呐–RGB 对比项，无需定制模型。

“取消所有对比损失”指本次融合 forward 中的独立全局对齐目标：
`loss_sonar_image`、`loss_sonar_text`。关闭开关后它们不会加入训练 loss 字典。
仍保留检测分类 FocalLoss、bbox/IoU、encoder/DN 检测监督、RGB prompt 和声呐类别热力图 BCE+Dice。
分类头虽使用视觉–文本点积，其检测分类目标不属于本次删除的独立对比目标；
删除它将连带改变检测器监督，不符合“其余部分不变”。
专用声呐预训练阶段的 ROI–文本损失不在融合 forward 中，本批配置不重跑或改写它。

## 声呐–RGB 对比项定义与 batch 注意事项

沿用仓库现有 ContrastiveLoss：声呐特征和 RGB encoder memory 分别做有效 token 均值池化，
L2 归一化后计算跨 batch 的相似度，使用双向交叉熵 InfoNCE，温度 0.07。
同一配对样本是正例，同一个进程的其它 batch 样本是负例，不是 ROI/逐像素对齐。
实现没有跨 GPU all_gather；每进程 batch size=1 时对比 loss 会退化为 0，
不能只靠梯度累积或增加 GPU 数恢复这个单步负例集合。配置保持原每进程 batch size=4。
同类/相似场景的不同样本也被作为负例，此处沿用现有定义，不额外设计标签感知负例。

## 每组文件和运行顺序

| 文件 | 初始化 | 训练 |
| --- | --- | --- |
| `rgbs50_finetune_20e.py` | 原 RGBS50 融合 `single_fusion_rgbs50/epoch_12.pth` | 20e，lr=5e-5，12/18 衰减 |
| `rgbs20_zeroshot.py` | 本组 RGBS50 finetune 的 `epoch_20.pth` | 保留用户指定 1 epoch，lr=5e-5，仅原 warmup |
| `rgbs20_finetune_12e.py` | 本组 RGBS50 finetune 的 `epoch_20.pth` | 12e，lr=5e-5，8/11 衰减 |

RGBS20 两种实验独立从 RGBS50 迁移，finetune 不接 zeroshot 结果。
所有组保持参考数据管线、batch size、优化器、梯度裁剪、类别设置和验证划分。
seed=42，resume=False，各自 work_dir 独立。若使用其它源 checkpoint，显式覆盖 load_from。
跨数据集六类→四类 DN label embedding 不匹配时，MMEngine 非严格加载会跳过该项重新初始化。

示例，仓库根目录、openmmlab 环境：

```bash
python tools/train.py configs/underwater_grounding_dino/oag_ablation/contrastive_loss/01_no_contrastive/rgbs50_finetune_20e.py
python tools/train.py configs/underwater_grounding_dino/oag_ablation/contrastive_loss/01_no_contrastive/rgbs20_zeroshot.py
python tools/train.py configs/underwater_grounding_dino/oag_ablation/contrastive_loss/01_no_contrastive/rgbs20_finetune_12e.py
```

添加声呐–RGB 组将路径中的 `01_no_contrastive` 替换成 `02_add_sonar_rgb`。

## 是否需要重新训练

无需从随机权重重训或重跑 RGB/声呐专用预训练：两组结构和参数名称完全相同，
可直接加载原融合 checkpoint。但对比损失只在训练时起作用，必须运行各自训练/微调，
仅改测试 config 而保持同一 checkpoint 不会产生该损失消融效果。

本批方案是“共同融合 checkpoint 初始化的微调阶段损失消融”。
无对比组保留参数仍可能已有对比学习历史，不能声称模型从未接受对比监督。
若要消除完整训练历史的对比监督，需相应重做前置 RGB/声呐和融合阶段，
并且给 baseline 统一匹配训练预算；这与本批固定预训练权重的方案是不同实验。

## 验证

已检查六份配置与参考模型的差异仅为三个对比损失开关/权重；数据加载器、
优化器与各训练 schedule 均保持一致（zeroshot 仍为 1e）。
使用 batch=2 缩小模型检查 loss/backward/predict；无对比组不输出独立对比 loss key，
添加组两项 loss 均非零，新增 loss 对 RGB 与声呐投影参数都有有限非零梯度。
未执行完整数据集训练、AP/收敛或 GPU/DDP 验证。
