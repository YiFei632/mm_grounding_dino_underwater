# 网页双先验热力图（更新）

`--web` 模式现在在选图后并排显示语义先验、声呐位置先验叠加图。
支持 OAGRGBPrompt / OAGStaged（fusion 阶段）以及 OAGAffine / OAGAffineQueryOnly 配置及匹配权重；
原始 OAG（不包含 RGBPrompt）没有这一声呐位置先验，网页会给出明确错误。
启动命令形式仍为 `python tools/analysis_tools/visualize_initial_queries.py CONFIG CHECKPOINT --web`。

两项数值均从真实 pre_decoder 执行中捕获，位于 top-k 选择之前：
语义使用 `enc_outputs_class.max(-1)` 原始 logit，声呐在 Affine 模型使用 `_position_scores` 未乘权重的响应，在 RGBPrompt/Staged 模型使用实际 `prompt_token_bias` 的 token 维最大值（未乘 sonar_position_weight）。
按 encoder 多尺度网格恢复分数图，屏蔽 padding/非有限值，双线性插值并取有效层平均，
裁掉 padding 后恢复原图分辨率。它表示 encoder 网格上的先验场，而非最终检测框或 query 点密度。
每张图独立 min/max 归一化显示，网页标注原始范围；颜色不能用来比较两项绝对大小。
支持调整透明度后点击“应用”、分别下载 PNG、下载包含 `semantic`/`sonar` 原始二维数组的 NPZ。
不带 `--web` 的离线 query 点导出功能保持原有行为。

下文为原始 query 点版本说明，其中网页绘点操作已由上述热力图界面替代。

# 验证集 Query 可视化网页

推荐在仓库根目录、openmmlab 环境启动：

```bash
python tools/analysis_tools/visualize_initial_queries.py \
  configs/underwater_grounding_dino/oag_ablation/00_baseline/rgbs50_finetune_20e.py \
  work_dirs/oag_train/single_fusion_rgbs50/epoch_12.pth \
  --web --device cuda:0 --host 127.0.0.1 --port 7860
```

用浏览器打开 **http://127.0.0.1:7860**。checkpoint 替换为实际要分析的权重。
模型启动时只加载一次，不预先推理整个验证集；点击图片才计算，最近 16 张结果缓存在内存中。
首张耗时取决于设备和原验证分辨率，不保证 GPU/CPU 的固定响应时间。
服务无需 Flask/Gradio 或联网加载前端资源；使用 Python 标准库和已有模型环境。

- 左侧仅列出验证集图片，支持文件名、图片 ID 或样本索引搜索及分页。
- 点击图片自动绘制 query，支持上一张/下一张；搜索只影响列表，前后切换按完整验证集索引。
- 显示全部点或抽样 50/100/300/600/900 个点，可实时调整半径、透明度和随机种子。
- 可切换原图/叠加图，点击图片切换原尺寸查看。
- 下载当前显示的原图分辨率 PNG、全部 query 坐标 NPZ、当前显示参数及抽样索引 JSON。
- PNG/JSON 由浏览器下载，NPZ 由服务端返回；网页模式不使用批量模式的 output-dir 写结果。
- 网页随机抽样使用稳定的浏览器伪随机顺序，相同种子可复现；不保证与命令行 NumPy 抽样索引相同。

远程机器使用：在模型所在机器启动服务，在自己电脑建立 SSH 转发：

```bash
ssh -L 7860:127.0.0.1:7860 用户名@服务器地址
```

随后在自己电脑打开 http://127.0.0.1:7860。
若需局域网直连，可启动时设置 `--host 0.0.0.0`，访问 `http://服务器IP:7860`。
服务没有登录认证，默认只绑定本机。按 Ctrl+C 停止服务并释放模型内存。

# 批量导出模式

在仓库根目录、openmmlab 环境执行：

```bash
python tools/analysis_tools/visualize_initial_queries.py \
  configs/underwater_grounding_dino/oag_ablation/00_baseline/rgbs50_finetune_20e.py \
  work_dirs/oag_train/single_fusion_rgbs50/epoch_12.pth \
  --device cuda:0 --num-images 5 \
  --output-dir work_dirs/initial_query_visualization/baseline
```

将 checkpoint 替换为实际要分析的权重即可；config 必须与 checkpoint 结构匹配，
脚本使用严格权重加载，不启动训练，也不加载 config 中的 `load_from`。
数据范围固定为 config 的 `val_dataloader.dataset`，不访问训练集或测试集。
沿用配置的完整数据管线，baseline 的真实配对声呐图也会被读取。

参数示例：

- `--indices 0 10 25`：指定 dataset 中从 0 开始的样本索引，不是 COCO img_id。
- `--start-index 20 --num-images 5`：处理第 20 到 24 个样本。
- `--max-points 100 --seed 42`：可复现地均匀随机抽样最多 100 个可见中心点；默认 0 画全部。
- `--point-radius 3 --alpha 200`：调整红点半径和不透明度。
- `--device cpu`：CPU 推理；默认优先 cuda:0。
- `--cfg-options key=value`：覆盖 config；正式对比建议保持原验证分辨率。

脚本在 eval 模式下，在 decoder 开始执行前捕获其实际 `reference_points`，
然后停止推理；不会将 encoder 网格位置、decoder 最终预测中心或 DN query 混入结果。
捕获值为各组实际进入 decoder 的初始框中心，包括 proposal 框回归对中心的更新。
固定 query 消融则得到其实际固定网格 reference boxes。

归一化中心先乘预处理后的 padded canvas 尺寸，再除 resize scale_factor 映射回原图。
padding 区域的中心不画，且不强行截断到图像边界；不支持 crop、rotation、flip、
TTA 或分块文本推理。使用标准验证 pipeline，batch size 固定为单图。

每个样本输出：

- `*_queries.png`：原始分辨率 RGB 图与红色中心点叠加结果。
- `*_queries.npz`：全部归一化 reference boxes、原图中心坐标、可见 mask 和实际绘制的 query 索引。
- `manifest.json`：本次 config/checkpoint、样本索引、RGB/声呐路径、图像尺寸及点数。

默认可视化全部 query；随机子集不代表 selector 分数最高的 query。
使用不同模型对比时保持相同 config 数据划分、indices、分辨率及绘制参数。
query 分布用于解释行为，不能代替检测 AP。

验证：现有 baseline 配置及 RGBS50 epoch_12 权重在 CPU 缩小输入下成功捕获并绘制 900 点；
另验证非正方形 padding/resize 坐标换算、padding 点排除及随机抽样可复现。
