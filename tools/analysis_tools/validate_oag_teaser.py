"""Verify teaser rendering with real OAG features and HTTP requests."""
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path("/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection")
sys.path.insert(0, str(ROOT / "tools/analysis_tools"))
import oag_teaser_web as teaser

args = SimpleNamespace(
    config=str(ROOT / "configs/underwater_grounding_dino/oag_ablation/"
               "contrastive_loss/02_add_sonar_rgb/rgbs20_finetune_12e.py"),
    checkpoint=str(ROOT / "work_dirs/oag_ablation/contrastive_loss/"
                   "02_add_sonar_rgb/rgbs20_finetune_12e/epoch_12.pth"),
    device="cuda:0", cfg_options=None, alpha=180, cache_size=2,
    repo_root=None)
state = teaser.load_state(args)
record = state.record(0)
assert record["family"] == "OAG"
image = state.original(0)
out = ROOT / "output/oag_teaser_checks"
out.mkdir(parents=True, exist_ok=True)
for category in ("0", "2", "all"):
    for level in ("all", "0", "3"):
        for kind in ("optical", "acoustic"):
            field, coverage = teaser.teaser_field(
                record, kind, category, level, "sigmoid")
            assert field.shape == (image.height, image.width)
            assert np.isfinite(field).all() and coverage.any()
for kind in ("optical", "acoustic"):
    field, coverage = teaser.teaser_field(record, kind, "0", "all", "sigmoid")
    low, high = float(field[coverage].min()), float(field[coverage].max())
    high = max(high, low + 1e-6)
    plain = teaser.render(
        image, field, coverage, kind, low, high, .5, 0)
    expected = image if kind == "optical" else ImageEnhance.Color(image).enhance(.25)
    assert np.array_equal(np.asarray(plain), np.asarray(expected))
    hidden = teaser.render(
        image, field, coverage, kind, high + 1, high + 2, .5, 180,
        transparent=True)
    assert np.asarray(hidden)[..., 3].max() == 0
    transparent = teaser.render(
        image, field, coverage, kind, low, high, .5, 180, transparent=True)
    assert transparent.mode == "RGBA" and transparent.size == image.size
    assert np.asarray(transparent)[..., 3].max() > 0
state.record(17)
assert "pre_decoder" not in state.model.__dict__
assert "forward" not in state.model.decoder.__dict__
server = teaser.create_server(state, "127.0.0.1", 0)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
base = "http://127.0.0.1:{}".format(server.server_port)


def fetch(path):
    with urlopen(base + path, timeout=120) as response:
        return response.read()


try:
    (out / "index.html").write_bytes(fetch("/"))
    (out / "info.json").write_bytes(fetch("/api/info"))
    (out / "images.json").write_bytes(fetch("/api/images"))
    (out / "meta.json").write_bytes(fetch("/api/meta/0"))
    (out / "original.png").write_bytes(fetch("/api/original/0"))
    for kind in ("optical", "acoustic"):
        query = "?kind={}&class=0&level=all&mode=sigmoid".format(kind)
        for suffix, extra in (
                ("", ""), ("_transparent", "&transparent=1"),
                ("_high_threshold", "&threshold=0.8"),
                ("_no_contours", "&contours=0")):
            payload = fetch("/api/render/0" + query + extra)
            preview = Image.open(BytesIO(payload))
            assert preview.size == image.size
            if "transparent" in suffix:
                assert preview.mode == "RGBA"
            (out / (kind + suffix + ".png")).write_bytes(payload)
        payload = fetch("/api/download/0" + query)
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            expected_field, expected_coverage = teaser.teaser_field(
                record, kind, "0", "all", "sigmoid")
            np.testing.assert_array_equal(archive["field"], expected_field)
            np.testing.assert_array_equal(archive["coverage"], expected_coverage)
            metadata = json.loads(str(archive["metadata_json"]))
            assert metadata["display"]["kind"] == kind
        (out / (kind + ".npz")).write_bytes(payload)
    assert (out / "acoustic.png").read_bytes() != (
        out / "acoustic_no_contours.png").read_bytes()
    for query in (
        "?threshold=nan", "?threshold=1.1", "?lower=2&upper=1",
        "?alpha=300", "?class=999", "?level=999", "?kind=unknown",
        "?saturation=-1",
    ):
        try:
            fetch("/api/render/0" + query)
            raise AssertionError("Bad parameters accepted")
        except HTTPError as error:
            assert error.code == 400
finally:
    server.shutdown()
    thread.join(5)
    server.server_close()
summary = dict(
    model=type(state.model).__name__, validation_count=len(state.images),
    config=args.config, checkpoint=args.checkpoint,
    classes=record["classes"], images_tested=[0, 17],
    geometry_all_classes_and_levels="passed",
    transparent_alpha_threshold_and_contours="passed",
    exported_fields_identical_to_model_capture="passed",
    http_endpoints_and_invalid_parameters="passed",
    persistent_server_started=False)
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
