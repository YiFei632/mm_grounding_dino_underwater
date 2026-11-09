# Query Selector 消融

基准：`oag_train/oag_double_resnet50_single_fusion_rgbs50_finetune_20e.py`。
仅改变 matching query 的位置初始化/排序；不删除任何模态、encoder/decoder 分支或损失。
`adapter.py` 是本目录注册的局部适配器，不修改 `mmdet` 源码。

## 三组定义

| 子目录 | 改动 | 保留内容 |
| --- | --- | --- |
| `01_fixed_queries` | 取消内容驱动的 query 排序/Top-K；900 个 matching query 的初始框固定为归一化 30×30 网格中心，宽高均为 0.05 | 原可学习 query 内容 embedding、全部 decoder、DN 训练 query、全部检测/辅助/提示/对比损失 |
| `02_no_text_prior` | 排序只用 `0.5 * sonar_prompt.max(class)`，去掉原分数里的 RGB–文本分类 logits | 文本 backbone/encoder/decoder 跨注意力、声呐模块、全部损失 |
| `03_no_sonar_prior` | 排序只用 `text_logits.max(token)`，去掉声呐加分项 | 声呐 backbone、映射提示、decoder 声呐注意力及其损失 |

原始排序为 `max_token(text_logits + 0.5 * prompt_token_bias)`。
无文本先验只改变 selector 的直接分类加分项。保留下来的声呐提示仍按原实现使用文本类别条件，
所以不能将本实验描述为“排序完全独立于文本”。完全移除这条间接依赖需要修改声呐 mapper，
会超出本次仅消融 selector 加分项、其余不变的范围。

固定位置是 decoder 的初始 reference boxes；不是把 900 个 content embedding 设为常量，
也不是禁止 decoder 后续逐层回归位置。DN query 不计入 900 个 matching query，训练时照常添加。
固定框不随输入/文本变化且不可学习，无新增可学习参数；宽高 0.05 为本实验明确的设计假设。

为保持原 encoder 辅助损失及其 tensor 数量，固定位置组在有效 encoder proposal 的索引顺序中
均匀采样 900 项，仅用于 encoder 分类/框回归辅助监督，不做按分数排序。
decoder 不使用这些 proposal 的位置或回归框，始终使用固定网格初始框。
该采样会随 feature map 有效尺寸变化，但 decoder 的归一化初始框不变。
若删除 encoder proposal 辅助监督则会额外改变损失，不符合本次约束。

## 九份配置及训练顺序

每个子目录提供以下三份：

| 文件 | 初始化 | 训练协议 |
| --- | --- | --- |
| `rgbs50_finetune_20e.py` | 原 `single_fusion_rgbs50/epoch_12.pth` | 与 baseline 相同：20e、lr=5e-5、500 iter warmup、12/18 衰减 |
| `rgbs20_zeroshot.py` | 本实验 RGBS50 finetune 的 `epoch_20.pth` | 严格保留用户设计的 1e 训练、lr=5e-5、500 iter warmup |
| `rgbs20_finetune_12e.py` | 同上；不接 zeroshot 输出 | 12e、lr=5e-5、500 iter warmup、8/11 衰减 |

所有配置 `num_queries=900`、`resume=False`、seed=42，输出目录相互独立。
RGBS20 类别、配对标注、数据管线、验证/测试设置均继承指定 RGBS20 参考。
RGBS20 仍有文本分支，可通过类别文本适配四类；DN label embedding 从六类到四类时，
MMEngine 非严格 checkpoint 加载会跳过形状不匹配项，不应据此对其它参数静默做截断。

必须先完成本组 RGBS50 训练，生成 `load_from` 指定文件，再独立运行两种 RGBS20 实验。
如实际 checkpoint 在别处，修改对应 `load_from`；不要加 `--resume` 来切换数据集。

示例（其余组替换目录名）：

```bash
python tools/train.py configs/underwater_grounding_dino/oag_ablation/query_selector/01_fixed_queries/rgbs50_finetune_20e.py
python tools/train.py configs/underwater_grounding_dino/oag_ablation/query_selector/01_fixed_queries/rgbs20_zeroshot.py
python tools/train.py configs/underwater_grounding_dino/oag_ablation/query_selector/01_fixed_queries/rgbs20_finetune_12e.py
```

## 是否需要重新训练

**无需重新进行前面的 RGB/声呐专用预训练，也无需从随机权重重训整个模型。**
三组均保留 baseline 的可学习参数及名称，固定网格为非持久 buffer，可以直接复用现有 checkpoint。
但改变 query 的位置分布/候选排名后，decoder 与检测头应重新适应，因此建议三组均执行上面的
RGBS50 20e 微调，再运行各自 RGBS20 实验；不建议仅替换推理配置作为训练消融结论。

这是“共同融合 checkpoint 初始化的 selector 微调阶段消融”。
若要研究“从 RGBS50 训练第一轮起就从未使用原 selector”的效果，
应从同一个声呐专用阶段 checkpoint 开始，给 baseline 及三个变体统一重跑 RGBS50 12e + 20e。
本批配置按用户指定的 RGBS50 finetune baseline 提供，没有额外生成这套更长的训练方案。

## 验证范围

配置解析、缩小模型 CPU loss/backward/predict 及合成分数扰动测试验证 selector 行为；
完整模型 checkpoint 兼容检查见执行结果。未执行实际数据集训练、收敛/AP 或 GPU/DDP 验证。
