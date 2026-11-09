"""Real-checkpoint checks for the standalone encoder response viewer."""

import argparse
import gc
import hashlib
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

ROOT = Path("/media/fishyu/fish-14tb-2/YiFei/Grounding_DINO/mmdetection")
sys.path.insert(0, str(ROOT / "tools/analysis_tools"))
sys.path.insert(0, str(ROOT))
from initial_query_response_web import (
    ViewerState, create_server, load_model_dataset, overlay,
    rasterize, select_values)
from visualize_initial_queries import capture_initial_queries


def main():
    output = ROOT / "output/encoder_response_viewer_checks"
    output.mkdir(parents=True, exist_ok=True)
    scenarios = [
        ("oag",
         "work_dirs/oag_ablation/contrastive_loss/02_add_sonar_rgb/"
         "rgbs20_finetune_12e/rgbs20_finetune_12e.py",
         "work_dirs/oag_ablation/contrastive_loss/02_add_sonar_rgb/"
         "rgbs20_finetune_12e/epoch_12.pth"),
        ("prism",
         "work_dirs/sonarbind_prism_double_resnet50_no_pretrain_finetune_20e_rgbs50/"
         "sonarbind_prism_double_resnet50_no_pretrain_finetune_20e_rgbs50.py",
         "work_dirs/sonarbind_prism_double_resnet50_no_pretrain_finetune_20e_rgbs50/"
         "epoch_1.pth"),
    ]
    summary = {}
    original_path = ROOT / "tools/analysis_tools/initial_query_web.py"
    original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    for name, config, checkpoint in scenarios:
        args = SimpleNamespace(
            config=str(ROOT / config), checkpoint=str(ROOT / checkpoint),
            device="cuda:0", cfg_options=None, alpha=180, cache_size=2)
        print("LOADING " + name, flush=True)
        model, dataset = load_model_dataset(args)
        state = ViewerState(model, dataset, args)
        record = state.record(0)
        reference, meta = capture_initial_queries(model, dataset[0])
        np.testing.assert_allclose(
            record["references_normalized"], reference, atol=1e-6, rtol=0)
        assert "forward" not in model.decoder.__dict__
        assert "pre_decoder" not in model.__dict__
        assert "gen_encoder_output_proposals" not in model.__dict__
        assert model.training is False
        if name == "oag":
            assert record["stages"] == ["encoder", "selection", "sonar"]
            np.testing.assert_allclose(
                record["arrays"]["selection_logit"],
                record["arrays"]["encoder_logit"] +
                record["arrays"]["sonar_logit"], atol=2e-5, rtol=1e-5)
        else:
            assert record["family"] == "Sonarbind-Prism"
            assert record["stages"] == ["encoder"]
        image = state.original(0)
        for mode in ("logit", "sigmoid"):
            for stage in record["stages"]:
                for category in ("all", "0"):
                    values = select_values(record, stage, category, mode)
                    for level in ("all", "0", str(len(record["shapes"]) - 1)):
                        field, coverage = rasterize(values, record, level)
                        assert field.shape == (image.height, image.width)
                        assert coverage.any() and np.isfinite(field).all()
        # A huge threshold and alpha=0 must both preserve the original RGB.
        values = select_values(record, "encoder", "0", "logit")
        field, coverage = rasterize(values, record)
        hi = float(values.max()) + 1
        assert np.array_equal(
            np.asarray(overlay(image, field, coverage, hi, hi + 1, 180)),
            np.asarray(image))
        assert np.array_equal(
            np.asarray(overlay(image, field, coverage, -100, 100, 0)),
            np.asarray(image))
        state.record(min(17, len(dataset) - 1))
        server = create_server(state, "127.0.0.1", 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = "http://127.0.0.1:{}".format(server.server_port)
        try:
            def fetch(path):
                with urlopen(base + path, timeout=120) as response:
                    return response.read()
            assert b"Encoder" in fetch("/")
            assert json.loads(fetch("/api/info"))["total"] == len(dataset)
            assert json.loads(fetch("/api/images"))["images"]
            data = json.loads(fetch("/api/queries/0"))
            (output / (name + "_metadata.json")).write_text(
                json.dumps(data, indent=2), encoding="utf-8")
            for endpoint in ("original", "points"):
                content = fetch("/api/{}/0".format(endpoint))
                assert Image.open(BytesIO(content)).size == image.size
                (output / (name + "_" + endpoint + ".png")).write_bytes(content)
            for stage in record["stages"]:
                content = fetch(
                    "/api/heatmap/0?stage={}&class=0&mode=logit&level=all"
                    .format(stage))
                assert Image.open(BytesIO(content)).size == image.size
                (output / (name + "_" + stage + ".png")).write_bytes(content)
            content = fetch("/api/download/0")
            with np.load(BytesIO(content), allow_pickle=False) as archive:
                assert archive["encoder_logit"].shape[1] == len(data["classes"])
                assert "metadata_json" in archive
            (output / (name + "_responses.npz")).write_bytes(content)
            bad_paths = [
                "/api/heatmap/0?class=999",
                "/api/heatmap/0?lower=2&upper=1",
                "/api/heatmap/0?lower=nan&upper=1",
                "/api/heatmap/0?level=999",
                "/api/heatmap/0?mode=unknown",
                "/api/queries/999999",
            ]
            if name == "prism":
                bad_paths.append("/api/heatmap/0?stage=selection")
            for path in bad_paths:
                try:
                    fetch(path)
                    raise AssertionError("Invalid request was accepted: " + path)
                except HTTPError as error:
                    assert error.code == 400, (path, error.code)
            summary[name] = dict(
                model=type(model).__name__, dataset_count=len(dataset),
                config=config, checkpoint=checkpoint,
                classes=data["classes"], stages=data["stages"],
                query_count=data["total_queries"],
                references_unchanged=True,
                geometry_threshold_and_api_checks="passed")
        finally:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()
        del record, state, model, dataset, server
        gc.collect()
        torch.cuda.empty_cache()
        print("PASSED " + name, flush=True)
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == original_hash
    summary["original_web_sha256"] = original_hash
    (output / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
