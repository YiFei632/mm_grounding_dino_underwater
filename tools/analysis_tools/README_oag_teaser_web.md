# OAG Teaser Viewer

New server files:

```text
/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/tools/analysis_tools/oag_teaser_web.py
/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/tools/analysis_tools/oag_teaser_web.html
```

Keep the existing `initial_query_response_web.py` in the same directory.
The new script reuses its capture and geometry implementation without
modifying it. Only OAG models with an actual RGB-space acoustic ranking
prior are supported: OAGRGBPrompt, OAGStaged fusion, and OAGAffine variants.
RGB-only and base OAG configs without this prior are rejected explicitly.

## Run

```bash
cd /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
/home/fishyu/anaconda3/envs/openmmlab/bin/python \
tools/analysis_tools/oag_teaser_web.py \
configs/underwater_grounding_dino/oag_ablation/contrastive_loss/02_add_sonar_rgb/rgbs20_finetune_12e.py \
work_dirs/oag_ablation/contrastive_loss/02_add_sonar_rgb/rgbs20_finetune_12e/epoch_12.pth \
--device cuda:0 --host 0.0.0.0 --port 7863
```

Replace the config and checkpoint with another matching supported OAG pair.
The viewer uses the config's validation split. No training is performed.
If the server port is inaccessible, create a client-side SSH forward:

```bash
ssh -N -L 17863:127.0.0.1:7863 fishyu@162.105.195.38
```

Then open `http://127.0.0.1:17863`. The validation run did not leave a
persistent service running.

## Panels

Optical: continuous turbo heatmap of the RGB-text encoder proposal-head
class response, before the sonar ranking bias. This is a class-response
visualization of the features, not raw feature channels or a post-decoder
feature map. The viewer does not claim an enhancement relative to another
checkpoint; a before/after study requires actual comparison features.

Acoustic: the actual weighted sonar class bias, or affine spatial prior,
rasterized in RGB coordinates. A pink field highlights high-response
locations over a partly desaturated RGB image. Up to three dark contour
lines show response levels. No Gaussian approximation or synthetic peak is
added. Affine variants have a class-agnostic prior.

Select validation image, class, and pyramid level. Each panel has independent
raw color bounds, opacity, and a visibility slider. Relative visibility
threshold `q` corresponds to `lower + q * (upper - lower)` in raw units.
Default acoustic threshold is 0.5. Increasing it hides weaker areas; it
does not change the underlying field. Absolute color bounds are editable.
Automatic bounds use valid token responses for the selected class.
All-level rasterization averages valid levels and is an inspection view.

Transparent background exports RGBA PNGs with no RGB base. This is useful
for placing the exact heatmap over the same RGB image in a figure editor.
PNG downloads keep the original image dimensions and omit UI labels.
NPZ downloads retain full raster fields, coverage masks, capture definitions,
class/level/score choices, color bounds and display settings.

## Checks

Real OAGStaged epoch-12 checkpoint, validation indices 0 and 17:
class/scale rasterization, original image geometry, threshold masking,
alpha-zero behavior, RGBA output, contour toggling, exact exported fields,
HTTP endpoints and invalid inputs passed.

Local browser checks use recorded real server API fixtures to check category
and threshold parameter propagation, transparent export controls, JavaScript
errors, and desktop/mobile layout. They do not rerun server inference.
Artifacts: `output/oag_teaser_checks` on the server and
`output/analysis/oag_teaser_checks` in the local workspace.
