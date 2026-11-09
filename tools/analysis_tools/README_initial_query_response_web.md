# Encoder Class Response Viewer

Server repository:

```text
/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection
```

New files, installed beside the original viewer:

- `tools/analysis_tools/initial_query_response_web.py`
- `tools/analysis_tools/initial_query_response_web.html`

The original `initial_query_web.py`, its HTML, and all model implementations
are preserved. The new script includes initial-query capture and geometry
helpers and can run against the separate original Grounding DINO repository.
It selects the model repository from the config path before importing mmdet.
An explicit `--repo-root` overrides that selection.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
/home/fishyu/anaconda3/envs/openmmlab/bin/python \
tools/analysis_tools/initial_query_response_web.py \
CONFIG CHECKPOINT --device cuda:0 --host 0.0.0.0 --port 7862
```

Replace CONFIG and CHECKPOINT with a matching model pair. Use another port
for a second instance. To compare models on the same validation images, both
configs must point to the same validation split and category prompt. The
browser displays the loaded model, config, and checkpoint.

The OAG instance launched on October 27, 2026 used server port 7862, physical
GPU 1, and `02_add_sonar_rgb/rgbs20_finetune_12e/epoch_12.pth`.
The tested local SSH forward is `http://127.0.0.1:17862`.
The server's public port was unreachable from the client. To recreate the
forward manually:

```bash
ssh -N -L 17862:127.0.0.1:7862 fishyu@162.105.195.38
```

That instance was confirmed stopped on October 28, 2026. Its PID/command and logs are under
`output/encoder_response_viewer_checks/running_viewer.json` and `web.log`.

## Original Grounding DINO

The exact requested model was verified on October 28, 2026. Run:

```bash
cd /media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdet-origin
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
/home/fishyu/anaconda3/envs/openmmlab/bin/python \
/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection/tools/analysis_tools/initial_query_response_web.py \
configs/mm_grounding_dino/grounding_dino_r50_finetune_12e_rgbs20.py \
work_dirs/grounding_dino_r50_finetune_12e_rgbs20/epoch_12.pth \
--device cuda:0 --host 0.0.0.0 --port 7862
```

The RGB-text encoder responses and initial query positions are available.
Sonar sections are hidden. Validation uses 1009 images from this config;
the previously tested OAG paired validation split contains 1004 images.
Align the validation split when comparing the same images.
The viewer does not change dataset settings automatically.

## Captured Values

- Grounding DINO: RGB-text encoder proposal-head responses before top-k.
- Sonarbind-Prism: the actual proposal classification head output after the
  fused encoder, before top-k selection.
- OAGRGBPrompt/OAGStaged: RGB-text encoder proposal classification, the actual
  weighted sonar token bias, and their sum used to rank initial queries.
- OAGAffine variants: classification plus the actual weighted, class-agnostic
  position prior. The UI explicitly labels that prior as class-agnostic.
- Base OAG and RGB-only stages without additive ranking: encoder responses
  and initial queries; an artificial sonar-added response is never generated.

OAG's prompt path adds scores, not RGB and sonar feature vectors. The tool
captures `prompt_token_bias` (or `_position_scores`) during normal model
execution and stops immediately before decoder attention. It restores all
temporary hooks afterward.

For a class spanning several text tokens, raw response is their mean logit.
The alternative sigmoid view is their mean sigmoid, matching the detector's
token-to-entity score convention. The class-average field is an inspection
view; actual query ranking still uses the model's maximum token score.
Sonar-only contribution always remains in additive logit units.

## Geometry and Display

Padding and invalid proposals are excluded. Each encoder level is interpolated
on the padded RGB canvas, cropped to the resized RGB image, then resized to
the original image. Selecting all levels averages valid levels; selecting
one level shows its native evidence without cross-level averaging.

Lower and upper display bounds are editable numbers in the selected score's
units. Values below the lower bound leave the RGB visible. Both main heatmaps
use the same color bounds. Automatic bounds use both stages' valid class
scores. The sonar contribution has separate bounds because it is an additive
term. Changing display controls uses cached inference.

Search, pagination, category and pyramid selection, original image, initial
query points, PNG downloads and raw NPZ downloads are available.

## Verification

Actual-checkpoint tests passed for OAGStaged (RGBS20, 1004 validation images)
and SonarBindPrism (RGBS50, 4479 validation images), inspecting two frames
per model. The checks cover class/scale rasterization, the additive identity,
original image dimensions, visibility thresholds, unchanged initial queries,
temporary hook cleanup, image/metadata/download endpoints and invalid input.
These checks establish compatibility; they are not a detection evaluation or
a fair comparison between those two different validation datasets.

The original GroundingDINO checkpoint `epoch_12.pth` passed inspection on
validation indices 0 and 17. Its captured references and per-class raw/sigmoid
responses matched ordinary full prediction. All four class maps, every
feature level, manual display threshold, PNG/NPZ output and HTTP endpoints
were checked. The imported mmdet package was confirmed to belong to
`mmdet-origin`. The temporary test server was closed after verification.

The live OAG UI also passed an Edge/Playwright check covering image selection,
category selection, sigmoid mode, manual lower bound, sonar contribution,
desktop/mobile rendering, and JavaScript errors. Screenshots are saved in the
local `output/analysis/encoder_response_viewer_checks` directory.

An older `utmdet_double_resnet50_zero_shot_rgbs20` config failed to build
against the server's current decoder implementation (`cross_attn_sonar_cfg`).
This existing config/code mismatch is not modified by the viewer.
