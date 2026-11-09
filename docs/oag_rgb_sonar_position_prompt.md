# RGB 坐标系中的声呐类别位置提示：方法与实现说明

本文对应新增配置 `configs/underwater_grounding_dino/oag/oag_double_resnet50_rgbs50_rgb_prompt.py`，模型类型为 `OAGRGBPrompt`。这是待实验验证的方法设计与实现说明，不包含精度提升、几何标定精度或跨域泛化效果的实验证据。

## 1. 问题定义与方法边界

输入为一对 RGB 图像与声呐图像，以及类别名称列表。两种传感器的视野、成像机理和目标投影不同，不假定像素对齐，也不假定相同归一化坐标代表同一物体。输出是 RGB 图像中的检测框及其类别。

目标是从当前声呐内容提取类别相关证据，利用 RGB–声呐特征的内容相关性，将这些证据组织到 RGB 特征网格，形成类别位置提示。提示在两处使用：

1. 对 Grounding DINO 的候选框排序增加有界偏置，改变初始 reference boxes 的选择。
2. 在每层 decoder 末尾，通过以 RGB 位置编码为 K/V 的注意力更新检测 query。

“位置映射”在本文中特指数据驱动的、样本依赖的软关联。它不是显式的相机–声呐投影模型，不输出外参或仿射矩阵，不保证一对一对应。类别框监督也不能唯一确定真实的跨模态对应矩阵。

## 2. 与 SCANet/SCAM 的关系

参考本地论文：Li 等，2024，*RGB-Sonar Tracking Benchmark and Spatial Cross-Attention Transformer Tracker*，第 IV-B 节、图 5、公式 (2)–(3)。

SCAM 将 RGB 与声呐各自的模板和搜索区域 token 合并为 $H_r,H_s$，以内容特征构造跨模态空间相关矩阵。原文写作：

$$
A_{r\leftarrow s}=\operatorname{ReLU}(Q_rK_s^\top/\sqrt C),\qquad
H_r^{attn}=A_{r\leftarrow s}V_r+H_r.
$$

反向分支类似。需要保留原文的细节：RGB 分支更新使用 $V_r$，声呐分支使用 $V_s$，并非通常的跨模态 K/V 同源写法。随后各分支使用独立的 Linear–GELU–Linear 模块和残差连接。ReLU 清除负相关项，不等价于严格识别或剔除所有背景。

本方法借鉴的是“不同模态的任意空间位置之间通过内容建立关联”，不复现 SCAM。本文将关联矩阵作用于声呐类别证据，而不是外观 value 特征；添加无匹配项并归一化；以 RGB 检测为任务，没有模板跟踪分支。论文写作时应称为受 SCAM 启发的独立设计，不能称为原 SCAM 的直接实现。

## 3. 记号和特征提取

| 记号 | 形状 | 含义 |
|---|---|---|
| $F^r$ | $B\times N_r\times D$ | 指定 RGB encoder 层级的特征 |
| $F^s$ | $B\times N_s\times D$ | 处理后的声呐局部内容特征 |
| $E$ | $B\times L\times D$ | encoder 输出的文本 token |
| $M$ | $B\times C\times L$ | 类别到文本 token 的归一化映射 |
| $T$ | $B\times C\times D$ | 类别文本向量 |
| $S$ | $B\times N_s\times C$ | 声呐类别证据 |
| $A$ | $B\times N_r\times N_s$ | 去除无匹配列后的关联权重 |
| $P$ | $B\times N_r\times C$ | RGB 类别位置提示 |

默认 $D=256$。RGB 经 ResNet50、四尺度 ChannelMapper、6 层 RGB–文本 encoder。映射使用第 2 层级（从 0 开始计数，对应 stride 32）。声呐灰度图复制成三通道，resize 至 $448\times448$，标准化后经独立 ResNet50 C5、FFT 低频过滤（半径比 0.6）、2048→256 线性投影，得到 $14\times14$ 网格。新增 Conv3×3–GroupNorm–GELU–Conv1×1 提取 $F^s$。声呐外观不进入 RGB–文本 encoder。

类别 token 映射复用检测模型的 tokenizer 与实体跨度定义：若第 $c$ 类有 $n_c$ 个有效 token，则这些 token 的 $M_{ct}=1/n_c$，其余为零。

$$T=ME.$$

这与对整段类别列表取一个全局均值不同。训练时类别顺序必须与 RGB、声呐标签一致；推理复用当前文本或当前 chunk 的实体 token 映射。空实体、没有有效 token 或截断后无法表达的类别显式报错。此实现要求显式类别/实体跨度，不支持 `tokens_positive=-1` 的整句无实体模式。

## 4. 声呐类别证据

对声呐与类别文本分别投影并 L2 归一化：

$$
z_{jc}=\frac{\langle\operatorname{norm}(W_{sc}F^s_j),
\operatorname{norm}(W_{tc}T_c)\rangle}{\tau}+b_c^{shared},\qquad
S_{jc}=\sigma(z_{jc}).
$$

实现中的偏置是所有类别共享的一个可学习标量，初始化为 -2；温度 $\tau=0.07$，固定。类别与空间位置都使用 sigmoid，允许同一位置与多个类别相关。填充类别与无效声呐位置的证据置零。这里的数值是可训练置信分数，并未经过概率校准。

此实现采用逐类别声呐框监督，替代旧 OAG 的单通道 Selector 监督和旧区域文本对比损失。声呐与文本之间仍然通过上述可学习相似度及监督建立语义联系，但不能将当前损失描述为额外的 InfoNCE/区域对比损失。

## 5. 含无匹配项的跨模态软映射

RGB 与声呐匹配使用独立于类别证据分支的投影：

$$r_{ij}=\langle\operatorname{norm}(W_rF^r_i),
\operatorname{norm}(W_sF^s_j)\rangle/\tau.$$

设有效声呐 token 数量为 $n_s$。关联 softmax 的真实位置 logits 为 $r_{ij}-\log\max(n_s,1)$，并增加一个可学习标量无匹配 logit $\nu$（初始化 0）：

$$
[A_{i1},\ldots,A_{iN_s},a_{i\varnothing}]
=\operatorname{Softmax}([r_{i1}-\log n_s,\ldots,r_{iN_s}-\log n_s,\nu]).
$$

无效声呐位置的 logit 为 $-\infty$。减去 $\log n_s$ 是为了避免真实位置仅因数量多就压倒单个无匹配项；它不影响真实位置彼此的相对概率。所有声呐位置都无效时，$a_{i\varnothing}=1$。

将声呐类别证据聚合到 RGB：

$$P_{ic}=\sum_j A_{ij}S_{jc}.$$

因此 $0\le P_{ic}\le 1-a_{i\varnothing}\le1$。无匹配质量不会被再次归一化掉，直接保留在提示幅度中。RGB padding 位置置零。$P$ 的行属于 RGB 网格，而 $S$ 的行属于声呐网格；这是本方法从声呐位置证据变为 RGB 位置提示的关键步骤。

这里没有强制几何邻近约束，匹配可以发生在任意位置。内容匹配可能受到重复纹理、同类多实例或模态域差异的干扰。辅助监督优化输出热图，并不唯一约束 $A$；因此注意力可视化不能作为几何对应正确性的充分证据。

## 6. 与 query selector 的融合

将低分辨率 $P$ 双线性插值到 RGB 四个特征层级，展平拼接为 $P^{pyr}$。插值遵循同一 RGB padded canvas 的网格比例，padding 候选被排除。

令 $U_{ct}=\mathbf1[M_{ct}>0]$，将类别提示广播至其文本 token：

$$
B_{it}=\frac{\sum_cP^{pyr}_{ic}U_{ct}}{\max(1,\sum_cU_{ct})}.
$$

分母处理实体 token 重叠，保持 $B\in[0,1]$。特殊 token、无效 token 和分类输出的额外 padding token 不获得偏置。设原分类 logits 为 $L_{it}$：

$$\tilde L_{it}=L_{it}+\alpha B_{it},\qquad
I=\operatorname{TopK}_i\max_t\tilde L_{it}.$$

默认 $\alpha=0.5$，$K=900$。无效 RGB proposals 被屏蔽；有效 proposals 少于 K 时显式报错。原 RGB 框回归分支提供候选框坐标，选出坐标 detach 后作为初始 reference boxes；query 内容仍来自可学习 query embedding。训练时保留 DN queries。

检测头的 encoder 分类监督使用选中位置的原始 $L$，而不是 $\tilde L$。Top-K 的索引操作不可微，提示模块不能仅靠排序获得梯度；第 8 节的独立提示损失以及 decoder 路径提供可微训练信号。

令 $\alpha=0$ 可以关闭提示排序偏置，但整个新模型不因此等价于旧 OAG（新 decoder 和辅助监督仍存在）。

## 7. decoder 中的 RGB 位置加强

每层执行常规 self-attention、query–text attention、query–RGB deformable attention 和 FFN 后，追加位置提示注意力。

当前 query $q_k$ 通过独立投影与类别文本计算类别混合权重：

$$
\pi_{kc}=\operatorname{Softmax}_c
\left(\langle\operatorname{norm}(W_{qc}q_k),
\operatorname{norm}(W_{dc}T_c)\rangle/0.07\right),
\qquad p_{ki}=\sum_c\pi_{kc}P_{ic}.
$$

这里是单独的语义混合分支，不直接复用检测头的分类 logits，也不读取 GT 类别来选择提示。DN query 与普通 query 使用相同计算。

由有效 RGB 网格生成正弦位置编码 $E_i^r$；query 的位置表示 $e_k$ 来自当前 RGB reference box。多头注意力每个头可写成：

$$
Q_k=W_Q(q_k+e_k),\quad K_i=W_KE_i^r,\quad V_i=W_VE_i^r,
$$

$$
\omega_{ki}=\operatorname{Softmax}_i
\left(Q_kK_i^\top/\sqrt{d_h}+\log\max(p_{ki},10^{-6})\right).
$$

声呐外观只用于生成 $P$，不直接作为该注意力的 K/V。设多头聚合和输出投影后为 $c_k$：

$$g_k=\max_{i\text{ valid}}p_{ki},\qquad
q_k^{out}=q_k+\beta g_k\operatorname{LayerNorm}(c_k).$$

默认 $\beta=0.1$。显式门控保留提示绝对幅度：若所有位置提示共同缩小，仅有 log-prior 的 softmax 会抵消公共缩放，而 $g_k$ 仍会缩小更新。$P=0$ 时更新严格为零；$\beta=0$ 时该新增注意力为恒等映射。低置信度通常意味着小更新，但 learned null 并不保证自动学到完美的缺失模态判别。

实现为完全 padding 的 RGB 网格加入仅在此情形可访问的零位置槽，避免全 $-\infty$ softmax；其提示门控为零。正常检测输入仍需足够的有效 RGB proposals。

## 8. 监督与优化目标

声呐 GT 框在原声呐画布坐标中栅格化为逐类别掩码 $Y^s_{jc}$。RGB GT 框经过训练数据增强后，以 `batch_input_shape` 对应的 padded canvas 栅格化为 $Y^r_{ic}$。一个格子与框有正面积交集就标为前景，多个同类框取并集。RGB padding 不参与损失。

设 $v^s_c$ 表示该样本声呐标注是否包含类别 c。RGB 提示训练目标为：

$$\hat Y^r_{ic}=Y^r_{ic}v^s_c.$$

这样，RGB 中存在而声呐未标注的类别不会被要求获得正的声呐提示；声呐存在而 RGB 中缺失的类别，RGB 目标热图为零。该规则依赖完整标注，并只是“共同类别可见”的代理：同类但不同实例同时出现时，它不能判断是否确为共同可见的同一目标。数据缺失标注会引入错误负样本，论文应披露此假设。

每种位置损失为 masked BCE + Dice。BCE 对有效位置和有效类别取均值；Dice 先按每个样本、每个类别在空间维求和，平滑项为 1，再对有效样本–类别对平均：

$$
\mathcal L_{dice}=\operatorname{Mean}_{b,c}
\left[1-\frac{2\sum_i m_{bic}p_{bic}y_{bic}+1}
{\sum_i m_{bic}(p_{bic}+y_{bic})+1}\right].
$$

声呐 BCE 使用 logits 版本，RGB BCE 使用截断至 $[10^{-6},1-10^{-6}]$ 的 P。默认配置使用 float32，注意力偏置在使用时转换至 query dtype。当前验证未覆盖外部 AMP OptimWrapper；启用混合精度前应单独检查关联矩阵、BCE 和梯度的数值稳定性。

$$
\mathcal L=\mathcal L_{GroundingDINO}
+\lambda_r\mathcal L_{pos}(P,\hat Y^r)
+\lambda_s\mathcal L_{pos}(\sigma(z),Y^s).
$$

默认 $\lambda_r=\lambda_s=1$。$\mathcal L_{GroundingDINO}$ 包含当前框架的最终层、辅助层、encoder 和去噪监督。基础 RGB prompt config 默认关闭旧 SonarBind 全局声呐–RGB/文本对比损失；专用 config `oag_double_resnet50_rgbs50_rgb_prompt_contrastive.py` 恢复 `loss_sonar_image` 和 `loss_sonar_text`，并将两个权重默认设为 0.25。它们通过 `model.loss_sonar_image_weight` 和 `model.loss_sonar_text_weight` 暴露，可直接在 config 中调整。新增的对比损失与类别位置监督同时优化。

所有 GT 仅用于训练损失，不用于构造前向的 S/A/P、Top-K 偏置或 decoder 类别混合。推理不要求声呐 GT。

## 9. 复杂度、实现路径和兼容性

低分辨率匹配主要开销为 $O(BN_rN_sD)$，关联矩阵空间为 $O(BN_rN_s)$。类别热图聚合为 $O(BN_rN_sC)$。每层位置注意力约为 $O(BQN_rD)$，常规多头实现会显式构造 $B\times H\times Q\times N_r$ 的 bias；应在完整分辨率和 batch size 下测量峰值显存，不能仅依据低分辨率测试认定可承受原 batch size。

新增文件：

- `mmdet/models/utils/sonar_rgb_prompt.py`：类别证据、无匹配关联、目标掩码与损失。
- `mmdet/models/detectors/oag_rgb_prompt.py`：encoder、类别 token 映射、Top-K 融合与损失集成。
- `mmdet/models/layers/transformer/oag_rgb_prompt_decoder.py`：RGB 位置注意力和显式可信度门控。
- `configs/underwater_grounding_dino/oag/oag_double_resnet50_rgbs50_rgb_prompt.py`：通过 custom_imports 注册新模型。
- `tests/test_models/test_detectors/test_oag_rgb_prompt.py`：数值、梯度、排序和模型回归检查。

旧配置和旧模型源文件保持原样；新模块通过新配置选择性导入。新模型有新增/替换的参数，不能把旧 checkpoint 严格恢复为完全相同的新模型。若使用旧 checkpoint 初始化，应检查 missing/unexpected keys，并重新训练提示模块；这与旧配置加载其原 checkpoint 的兼容性是两件事。

训练入口：

```bash
python tools/train.py configs/underwater_grounding_dino/oag/oag_double_resnet50_rgbs50_rgb_prompt.py
```

## 10. 推荐实验与论文表述

至少比较：原始 OAG；只开新 selector（beta=0）；只开新 decoder（alpha=0）；两者全开。进一步消融无匹配项、幅度门控、类别条件化、RGB 提示监督及映射层级。为区分“新提示机制”与“额外监督”的贡献，应设置共享监督但关闭对应注入路径的控制组。

建议报告 RGB 检测 AP、共同可见/单模态可见分组表现、初始 proposal recall、运行时间与峰值显存。对 P 做空间可视化，对 A 做关联可视化，并包括同类多实例、声呐空目标、RGB 空目标与低反射等失败案例。若要论证几何/实例对应，必须额外提供相应标注或独立评估。

可使用的方法表述：

> 我们构建类别条件化的声呐空间证据，并通过具有无匹配选项的跨模态内容关联将其聚合到 RGB 特征网格。得到的 RGB 类别位置提示以残差偏置参与候选查询选择，并通过置信度门控的位置注意力在解码过程中持续增强检测查询。该方法在无显式几何标定的条件下学习任务驱动的空间软关联。

在实验完成前，不应写成“准确恢复跨模态几何关系”“保证实例级对应”或“显著提高检测精度”。

## 11. 已执行的验证及其范围

验证环境为本机 `openmmlab` Python 环境、PyTorch 2.2.0，使用 CPU。NVIDIA 驱动当前不可用。以下是功能检查，不是数据集精度实验：

| 检查 | 结果 | 范围 |
|---|---|---|
| 类别提示、padding、全无效声呐、无匹配项及梯度 | 通过 | 随机小张量 |
| 类别提示改变 Top-K 排序、特殊 token 不加偏置 | 通过 | 可控分数例子 |
| 零提示恒等更新、全 padding 有限输出、非零提示梯度 | 通过 | 独立 decoder 提示模块 |
| 框栅格化使用 RGB padded canvas | 通过 | 已知框坐标 |
| 原 position-only OAG 与新模型 loss/backward/predict | 通过 | 1 层 encoder/decoder、10 queries、合成 RGB128、声呐64 |
| 新模型空声呐标注训练与反向传播 | 通过 | 同上 |
| 原配置与新配置的完整结构初始化和推理 | 通过 | 原样模型配置，6 层、900 queries、声呐448，合成 RGB256，B=1，无 GT 推理 |
| 新配置完整结构 loss/backward | 通过 | 原样模型配置，6 层、num_cp=6、900 queries、声呐448，合成 RGB256，B=1 |

回归测试文件支持 pytest 收集，也可在没有安装 pytest 时直接执行：

```bash
PYTHONPATH=. python tests/test_models/test_detectors/test_oag_rgb_prompt.py
```

该入口运行前六项检查。完整结构检查额外加载了配置指定的 backbone 预训练权重，并检查了所有映射模块参数和各层位置注意力的梯度。未执行完整 RGBS50 训练、多卡 DDP、GPU/AMP 或原训练分辨率与 batch size 下的显存测试。旧配置文件及旧 OAG/decoder 实现没有因本次功能而改动。
