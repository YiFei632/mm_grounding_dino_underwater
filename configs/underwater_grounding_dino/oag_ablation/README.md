# 模态与模态处理消融

以 `oag_train/oag_double_resnet50_single_fusion_rgbs50_finetune_20e.py`
的实际配置与实现为基准。所有新增内容均在本目录；不修改 `mmdet` 源码或原配置。

## 目录及结构

| 子目录 | 实验 | 实现 |
| --- | --- | --- |
| `00_baseline` | 原始 RGB + 文本 + 声呐 + FFT | 原样继承指定微调配置，仅改输出目录并固定随机种子 |
| `01_no_sonar` | RGB + 文本 | 现有 GroundingDINO；不构造声呐 backbone、投影、selector、RGB prompt mapper 或 decoder 声呐提示注意力；query 仅按 RGB–文本分类分数选择 |
| `02_no_text` | RGB + 声呐 | 本地适配器；无 BERT/文本投影、encoder 文本层/融合层、decoder 文本 cross-attention；query 仅按映射后的声呐先验选择 |
| `03_no_fft` | RGB + 文本 + 声呐，关闭 FFT | `model.fft_band0_filter.enabled=False`；FFT 模块直接返回输入，不执行 FFT/IFFT |

所有实验保持相同 RGBS50 六类、RGB 图像划分、RGB 数据增强、batch size=4、
AdamW、梯度裁剪、学习率倍率、验证设置及随机种子 42。
保留分支的损失权重与 baseline 一致；删除模态时删除依赖该模态的损失。
单个 seed 不足以估计实验方差，正式结果可给所有组使用同一组多个 seed。

### 无声呐

通过 `_delete_=True` 重建为现有 `GroundingDINO` 配置，不只是将损失权重置零。
RGB backbone/neck、RGB–文本 encoder、query embedding、检测头和 DN 设置均来自 baseline。
数据管线移除 `LoadSonarImage`、`LoadSonarAnnotations` 和声呐元信息，
但不改变 RGB 样本集。baseline 的 `sonar_selector` 已被 RGB prompt mapper 替代，
本实验将这整条提示分支移除。

### 无文本：为什么必须有本地适配代码

现有 `OAGRGBPrompt` 构造函数明确禁止关闭文本或语言引导 query；
其分类、声呐类别图、RGB prompt 及 decoder prompt 类别混合都需要类别表示。
因此不能用 `use_text_branch=False` 或 `language_model=None` 两个开关直接完成。

`no_text_adapter.py` 通过 `custom_imports` 注册 `OAGNoTextAblation`：

- 不构造 BERT 或文本投影；encoder 为纯 RGB 多尺度可变形自注意力及 FFN。
- 用六个随机初始化、可学习的类别向量代替语言语义。
  原 GroundingDINOHead 的点积分类运算成为固定类别原型分类器，无需改写匹配与检测损失。
  类别序号严格沿用 RGBS50 的 `metainfo.classes`；不读取类别名称内容或分词。
- 声呐类别图及映射到 RGB 的提示仍沿用 baseline 的计算与独立声呐 GT 监督，
  但类别条件来自学习向量。内部 `memory_text`、`text_token_mask`、`text_class`
  是复用接口/参数名，输入是类别向量，既无文本输入也无文本编码。
- query 排序为映射后声呐先验在类别维度的最大值；不叠加 RGB 分类 logits。
  无效/填充 proposal 被屏蔽。参考框仍来自 RGB proposal 回归，
  因为声呐与 RGB 使用各自坐标，不能直接把声呐框当作 RGB 框。
- 删除 decoder 文本 cross-attention 和对应 LayerNorm；保留 RGB cross-attention、
  自注意力、FFN 及声呐空间提示注意力。加载 baseline 权重时自动重映射剩余 norm 索引；
  加载本实验自己的 checkpoint 不重映射。
- 关闭声呐–文本全局对比损失；保留 RGB prompt 和声呐类别监督。

该实验是六类闭集检测，失去开放词汇能力；无法将其描述为保留全部开放词汇能力的无文本模型。
移除文本后为分类提供非文本类别表示是必要适配，同时也是报告结果时应披露的结构差异。

## 是否需要重新预训练

区分“能否复用权重”和“训练历史是否满足消融定义”。

| 实验 | 技术上必须重做外部预训练？ | 推荐方案与限制 |
| --- | --- | --- |
| baseline | 否 | 使用原 RGBS50 12e 权重微调 20e |
| 无声呐 | 否 | 可加载保留的 RGB/文本参数，微调 20e；正式比较建议本结构重训 RGBS50 12e + 20e |
| 无文本 | 否，但新增类别向量必须训练 | 不能只删分支直接测 AP；建议重训 RGBS50 12e + 20e，让类别向量及提示分支重新适应 |
| 无 FFT | 否，FFT 无可学习参数 | 可直接关闭后微调；建议重训 RGBS50 12e + 20e，以适应未滤波特征分布 |

**共同融合权重微调**回答“在相同已训练融合模型上删去该组件，再适应 20e 的影响”。
这是一种明确的微调阶段消融；保留参数可能已经从被删除模态获益。

**RGBS50 从头重训 12e + 微调 20e**回答“使用相同外部预训练初始化，
从 RGBS50 第一轮就采用该消融结构的效果”。各组总训练轮数相同。
这里“从头”只指 RGBS50 阶段，不表示随机初始化所有权重。

如果论文主张“训练历史中完全没有某模态/FFT”，上述两种方案都不够：

- 无声呐：应从声呐阶段之前的 `rgb_duo/epoch_12.pth` 出发，跳过 UATD/MDSD/SCTD，
  再训练无声呐 RGBS50 12e + 20e。可覆盖 `rgbs50_train_12e.py` 的 `load_from`。
  与共享预训练方案相比，此方案还改变了预训练数据，应单独报告。
- 无文本：原 RGB、声呐阶段均有文本监督。要声称全过程无文本，必须重新设计并训练
  无文本的 RGB/声呐专用阶段（包括各数据集类别空间和相应分类/热力图监督）。
  本次提供的是固定 RGBS50 六类的融合阶段适配器，不能直接拿它运行原声呐专用阶段；
  也不应将加载既有权重的实验标为“全流程无文本预训练”。
- 无 FFT：至少从 `rgb_duo/epoch_12.pth` 开始，重跑关闭 FFT 的 UATD → MDSD → SCTD
  和 RGBS50 12e + 20e；RGB 阶段不经过 FFT，无需重跑。
  本次 `03_no_fft` 只关闭 RGBS50 阶段 FFT，未自动重建三个声呐专用预训练阶段。

建议首轮采用所有组相同的 **RGBS50 12e + 20e** 预算，并明确写“固定外部预训练的融合阶段消融”。
若只是快速验证，可使用所有组共同融合权重的单独 20e 方案。
不要混合两种方案的结果进行公平性比较。

## 每个子目录中的三个配置

| 文件 | 初始化 | 训练 |
| --- | --- | --- |
| `rgbs50_finetune_20e.py` | 原 `work_dirs/oag_train/single_fusion_rgbs50/epoch_12.pth` | 20e，lr=5e-5，12/18 衰减 |
| `rgbs50_train_12e.py` | 原 `work_dirs/oag_train/sonar_sctd_coco/epoch_12.pth` | 此实验结构重训 RGBS50 12e，lr=1e-4，8/11 衰减 |
| `rgbs50_retrain_finetune_20e.py` | 本子目录 `rgbs50_train_12e` 输出 | 再训练 20e，lr=5e-5，12/18 衰减 |

每阶段重新建立 optimizer，均 `resume=False`。切换阶段不要加 `--resume`。
`work_dir` 相互独立；若命令行覆盖输出目录，应同步修改下一阶段 `load_from`。

## 启动示例

在仓库根目录和 `openmmlab` 环境中执行。推荐两步方案，以无文本为例：

```bash
python tools/train.py configs/underwater_grounding_dino/oag_ablation/02_no_text/rgbs50_train_12e.py
python tools/train.py configs/underwater_grounding_dino/oag_ablation/02_no_text/rgbs50_retrain_finetune_20e.py
```

其余组替换 `02_no_text` 为对应目录即可。快速 20e 方案：

```bash
python tools/train.py configs/underwater_grounding_dino/oag_ablation/02_no_text/rgbs50_finetune_20e.py
```

训练启动时，已删除分支的 checkpoint keys 被忽略是预期行为。
无文本会提示 `category_vectors.weight` 缺失并使用随机初始化。
从 SCTD 专用 checkpoint 切到六类 RGBS50 时，旧配置实际使用三类 DN embedding，
可能出现 DN embedding 形状不匹配，MMEngine 非严格加载会跳过该参数；
原 staged 配置注释中的“所有阶段统一 10 类”与实际逐数据集覆盖值不同。
不要把其它未解释的关键参数缺失当作正常情况。

## 已完成验证

- 12 个训练配置均通过 MMEngine 继承解析与训练/验证管线绑定检查。
- 四个完整深度模型均可加载现有 RGBS50 `epoch_12.pth`：
  baseline/无 FFT 完全匹配；无声呐保留参数无缺失，舍弃 397 个 tensor；
  无文本只缺新增类别向量，舍弃 379 个 tensor。
- 无文本 decoder LayerNorm 权重重映射正确；其自身 state_dict 可 strict=True 重载。
- 四个缩小模型通过 CPU loss、backward、predict，所有 loss/输出有限。
- 无声呐模型不包含声呐/提示参数或声呐加载步骤；无文本模型没有语言模型、
  文本 encoder 或 decoder 文本 cross-attention；类别向量与提示映射得到有效梯度。
- 无文本空声呐 GT 情况可反向传播；关闭 FFT 后输出与输入逐元素一致。

可复核：

```bash
python configs/underwater_grounding_dino/oag_ablation/checks/smoke.py
python configs/underwater_grounding_dino/oag_ablation/checks/checkpoint.py
```

尚未执行完整数据集训练或 GPU/DDP 验证，验证结果不代表收敛或 AP。

## RGBS20：无声呐与无文本

`01_no_sonar`、`02_no_text` 中各新增：

- `rgbs20_zeroshot.py`：严格沿用指定参考的 **1 epoch 训练协议**、500 iteration warmup、lr=5e-5。
- `rgbs20_finetune_12e.py`：沿用指定参考的 12 epoch、8/11 衰减、lr=5e-5。

默认加载对应组 `rgbs50_finetune_20e/epoch_20.pth`，而非完整多模态 baseline；
若实际使用 12e→20e 重训路线，请将 `load_from` 改为该组
`rgbs50_retrain_finetune_20e/epoch_20.pth`。每个实验输出目录独立。
RGBS20 四类、配对标注、声呐布局及增强均继承 `oag_train` 参考配置。
无声呐组额外去除声呐图像/标注加载和声呐元信息。

无文本组使用 `rgbs20_no_text_adapter.py`，只在源/目标类别向量形状不同时舍弃源向量，
随机初始化四个 RGBS20 类别向量；继续加载自己的四类 checkpoint 则保留已训练向量。
其余保留结构沿用 RGBS50 无文本版本，FFT 保持开启。
6→4 类的 DN embedding 由 MMEngine 非严格加载跳过不匹配参数并重新初始化。
此组仍保留用户指定的 zeroshot 文件名及一轮训练设计，但无文本模型是闭集分类器，
其新类别向量不具备语言驱动的零样本语义迁移能力，应在解释实验结果时明确这一点。

## 仅 ResNet50 预训练初始化的独立实验（2026-10-22 更新）

按最新要求，`01_no_sonar/rgbs50_train_12e.py` 已改为 `load_from=None`，
并新增 `01_no_sonar/rgbs20_train_12e.py`。两者分别在对应数据集独立训练，
不串联 RGBS50→RGBS20，不加载 OAG/声呐阶段或检测 checkpoint。
两份配置均为 12 epoch、AdamW lr=1e-4、500 iter warmup、8/11 衰减。

唯一保留的预训练参数是 RGB backbone 的本地 `checkpoints/resnet50-0676ba61.pth`。
保留原 GroundingDINO 文本分支与结构；`resnet_only_init.py` 中的
`RandomInitAblationBert` 在原包装器构造后重新初始化所有 BERT 层（包括词嵌入），
不保留 BERT 预训练权重，但仍使用现有 tokenizer/词表。
neck、encoder、decoder、检测头及 query 模块均使用各自默认初始化。
原参数学习率倍率保留，BERT lr_mult=0.01；随机 BERT/检测模块训练 12e 不保证收敛，
本实验不能描述为“整个检测系统都只是在微调已有检测能力”。

为避免覆盖旧实验，新输出目录分别为：

- `work_dirs/oag_ablation/01_no_sonar/rgbs50_resnet_only_train_12e`
- `work_dirs/oag_ablation/01_no_sonar/rgbs20_resnet_only_train_12e`

旧配置如 `rgbs50_retrain_finetune_20e.py` 或 `rgbs20_finetune_12e.py` 的
`load_from` 仍指向其原有实验目录，不会自动改为本次独立训练输出。
若需要继续训练本次结果，应显式将对应 `load_from` 改为上述新目录中的 checkpoint。
不要使用 `--resume` 或命令行 `--cfg-options load_from=...` 启动本次独立初始化实验。
本节覆盖前文对 `01_no_sonar/rgbs50_train_12e.py` 原初始化与输出目录的说明。
