"""Check the exact original Grounding DINO checkpoint and standalone viewer."""

from io import BytesIO
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch

BASE = Path("/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO")
ROOT = BASE / "mmdet-origin"
TOOLS = BASE / "mmdetection/tools/analysis_tools"
sys.path.insert(0, str(TOOLS))
from initial_query_response_web import (
    ViewerState, create_server, load_model_dataset, overlay,
    rasterize, select_values)

args = SimpleNamespace(
    config=str(ROOT / "configs/mm_grounding_dino/"
               "grounding_dino_r50_finetune_12e_rgbs20.py"),
    checkpoint=str(ROOT / "work_dirs/grounding_dino_r50_finetune_12e_rgbs20/"
                   "epoch_12.pth"),
    cfg_options=None, device="cuda:0", alpha=180, cache_size=2)
model, dataset = load_model_dataset(args)
import mmdet
from mmengine.dataset import pseudo_collate

assert ROOT in Path(mmdet.__file__).resolve().parents
state = ViewerState(model, dataset, args)
record = state.record(0)
assert record["family"] == "Grounding-DINO"
assert record["stages"] == ["encoder"]
assert record["classes"] == ["fish", "fishing_net", "mine", "stick"]
assert not hasattr(model, "use_text_branch")
assert "pre_decoder" not in model.__dict__
assert "forward" not in model.decoder.__dict__
captured = {}


def references(module, inputs, kwargs):
    captured["references"] = kwargs["reference_points"][0].detach().cpu().numpy()


def classification(module, inputs, output):
    captured["logits"] = output[0].detach().float().cpu()


handle = model.decoder.register_forward_pre_hook(references, with_kwargs=True)
head_handle = model.bbox_head.cls_branches[
    model.decoder.num_layers].register_forward_hook(classification)
try:
    with torch.inference_mode():
        predictions = model.test_step(pseudo_collate([dataset[0]]))
    assert len(predictions) == 1
finally:
    handle.remove()
    head_handle.remove()
np.testing.assert_allclose(
    captured["references"], record["references_normalized"], atol=1e-6, rtol=0)
for category, indices in enumerate(record["token_indices"]):
    expected = captured["logits"][:, indices]
    np.testing.assert_allclose(
        expected.mean(-1).numpy(),
        record["arrays"]["encoder_logit"][:, category], atol=1e-5)
    np.testing.assert_allclose(
        expected.sigmoid().mean(-1).numpy(),
        record["arrays"]["encoder_sigmoid"][:, category], atol=1e-6)
image = state.original(0)
for mode in ("logit", "sigmoid"):
    for category in ("all", "0", "1", "2", "3"):
        for level in ("all", "0", "1", "2", "3"):
            field, coverage = rasterize(
                select_values(record, "encoder", category, mode),
                record, level)
            assert field.shape == (image.height, image.width)
            assert np.isfinite(field).all() and coverage.any()
field, coverage = rasterize(
    select_values(record, "encoder", "0", "logit"), record)
high = float(field.max()) + 1
assert np.array_equal(np.asarray(image), np.asarray(
    overlay(image, field, coverage, high, high + 1, 180)))
second = state.record(17)
assert second["family"] == "Grounding-DINO"
server = create_server(state, "127.0.0.1", 0)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
out = BASE / "mmdetection/output/origin_response_viewer_checks"
out.mkdir(parents=True, exist_ok=True)
base = "http://127.0.0.1:{}".format(server.server_port)


def fetch(path):
    with urlopen(base + path, timeout=60) as response:
        return response.read()


try:
    (out / "index.html").write_bytes(fetch("/"))
    info = json.loads(fetch("/api/info"))
    assert info["repo_root"] == str(ROOT)
    assert json.loads(fetch("/api/images"))["images"]
    metadata = json.loads(fetch("/api/queries/0"))
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    for path, name in [
        ("/api/original/0", "original.png"),
        ("/api/points/0", "queries.png"),
        ("/api/heatmap/0?stage=encoder&class=0", "encoder.png"),
        ("/api/heatmap/0?class=2&mode=sigmoid&lower=0.1&upper=1",
         "mine_sigmoid.png"),
    ]:
        payload = fetch(path)
        assert Image.open(BytesIO(payload)).size == image.size
        (out / name).write_bytes(payload)
    payload = fetch("/api/download/0")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        assert archive["encoder_logit"].shape[1] == 4
        assert "selection_logit" not in archive
    (out / "responses.npz").write_bytes(payload)
    for path in [
        "/api/heatmap/0?stage=sonar",
        "/api/heatmap/0?stage=selection",
        "/api/heatmap/0?lower=2&upper=1",
    ]:
        try:
            fetch(path)
            raise AssertionError("Invalid request was accepted.")
        except HTTPError as error:
            assert error.code == 400
finally:
    server.shutdown()
    worker.join(5)
    server.server_close()
summary = dict(
    model=type(model).__name__, repo_root=args.repo_root,
    imported_mmdet=mmdet.__file__, config=args.config,
    checkpoint=args.checkpoint, validation_count=len(dataset),
    classes=record["classes"], num_queries=len(record["visible"]),
    stages=record["stages"], full_predict_reference_comparison="passed",
    full_predict_class_score_comparison="passed",
    all_class_and_scale_rasterization="passed",
    threshold_and_http_checks="passed", tested_image_indices=[0, 17],
    persistent_server_started=False)
(out / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
