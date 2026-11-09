#!/usr/bin/env python
"""OAG-only optical class-response and acoustic-prior teaser browser.

Keep initial_query_response_web.py beside this script. Model capture and
geometry use that implementation; original scripts and model code are intact.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageEnhance

import initial_query_response_web as base

RENDER_LOCK = threading.Lock()


def teaser_field(record, kind, category, level, mode):
    if kind not in ("optical", "acoustic"):
        raise ValueError("Unknown teaser panel.")
    stage = "encoder" if kind == "optical" else "sonar"
    values = base.select_values(record, stage, category, mode)
    return base.rasterize(values, record, level)


def render(image, field, coverage, kind, lower, upper, threshold, alpha,
           saturation=0.25, contours=True, transparent=False):
    """Render actual fields; threshold is relative to explicit color bounds."""
    if kind not in ("optical", "acoustic"):
        raise ValueError("Unknown teaser panel.")
    if not np.isfinite([lower, upper, threshold, alpha, saturation]).all():
        raise ValueError("Display parameters must be finite.")
    if not lower < upper or not 0 <= threshold <= 1:
        raise ValueError("Expected lower < upper and threshold in [0, 1].")
    if not 0 <= alpha <= 255 or not 0 <= saturation <= 1:
        raise ValueError("Invalid alpha or background saturation.")
    if field.shape != (image.height, image.width) or coverage.shape != field.shape:
        raise ValueError("Field dimensions do not match the original RGB.")
    t = np.clip((field - lower) / (upper - lower), 0, 1)
    valid = coverage & np.isfinite(field)
    visible = valid & (t >= threshold)
    background = Image.new("RGBA", image.size) if transparent else (
        ImageEnhance.Color(image).enhance(
            saturation if kind == "acoustic" else 1).convert("RGBA"))
    with RENDER_LOCK:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import colormaps
        if kind == "optical":
            rgb = (colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)
            opacity = np.where(visible, alpha, 0).astype(np.uint8)
        else:
            rgb = np.zeros((*t.shape, 3), dtype=np.uint8)
            rgb[...] = [232, 57, 85]
            opacity = np.where(visible, alpha * t, 0).astype(np.uint8)
        layer = Image.fromarray(np.dstack((rgb, opacity)), "RGBA")
        result = Image.alpha_composite(background, layer)
        if kind == "acoustic" and contours and alpha > 0:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            # Contour vertices use exact image-pixel centers; no tight bbox crop.
            h, w = field.shape
            fig = Figure(figsize=(w / 100, h / 100), dpi=100)
            fig.patch.set_alpha(0)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.patch.set_alpha(0)
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_axis_off()
            data = np.ma.array(t, mask=~valid)
            # Keep fixed relative levels; skip levels absent in this field.
            levels = [float(v) for v in np.linspace(threshold, 1, 5)[1:4]
                      if valid.any() and t[valid].min() < v < t[valid].max()]
            if levels:
                ax.contour(data, levels=levels, colors=["#9e1235"],
                           linewidths=1.2, alpha=alpha / 255)
                canvas.draw()
                lines = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy())
                result = Image.alpha_composite(result, lines)
            fig.clear()
    return result if transparent else result.convert("RGB")


class TeaserState(base.ViewerState):
    def record(self, index):
        record = super().record(index)
        if record["family"] != "OAG" or "sonar" not in record["stages"]:
            raise ValueError(
                "Use an OAG config that executes a RGB-space sonar ranking "
                "prior (OAGRGBPrompt/OAGStaged fusion or OAGAffine).")
        return record

    def metadata(self, index, record):
        data = super().metadata(index, record)
        data["optical_definition"] = (
            "Class response of RGB-text encoder features before sonar ranking "
            "score addition; not raw feature channels or post-decoder features.")
        data["acoustic_definition"] = (
            "Actual weighted sonar contribution in RGB proposal coordinates; "
            "display thresholds and contours do not change model values.")
        return data


def create_server(state, host, port):
    html = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, mime, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition",
                                 'attachment; filename="{}"'.format(filename))
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value, status=200):
            self.send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                      "application/json; charset=utf-8", status)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == "/":
                    return self.send(html, "text/html; charset=utf-8")
                if url.path == "/api/info":
                    return self.json(dict(
                        total=len(state.images), model=type(state.model).__name__,
                        config=state.args.config, checkpoint=state.args.checkpoint))
                if url.path == "/api/images":
                    term = q.get("q", "").casefold()
                    found = [v for v in state.images if term in v["name"].casefold()
                             or term in (str(v["index"]), v["img_id"].casefold())]
                    offset = max(0, int(q.get("offset", 0)))
                    return self.json(dict(total=len(found),
                                          images=found[offset:offset + 40]))
                if not url.path.startswith((
                        "/api/meta/", "/api/render/", "/api/original/",
                        "/api/download/")):
                    return self.json(dict(error="Not found"), 404)
                index = int(url.path.rsplit("/", 1)[1])
                state.validate(index)
                image = state.original(index)
                buffer = BytesIO()
                if url.path.startswith("/api/original/"):
                    image.save(buffer, format="PNG")
                    return self.send(buffer.getvalue(), "image/png")
                record = state.record(index)
                if url.path.startswith("/api/meta/"):
                    return self.json(state.metadata(index, record))
                kind = q.get("kind", "optical")
                category, level = q.get("class", "0"), q.get("level", "all")
                mode = q.get("mode", "sigmoid")
                field, coverage = teaser_field(
                    record, kind, category, level, mode)
                values = field[coverage]
                if not len(values):
                    raise ValueError("No valid pixels in selected feature level.")
                lower = float(q.get("lower", values.min()))
                upper = float(q.get("upper", values.max()))
                if "upper" not in q and upper <= lower:
                    upper = lower + max(1e-6, abs(lower) * 1e-6)
                threshold = float(q.get("threshold", 0.5 if kind == "acoustic" else 0))
                alpha = int(q.get("alpha", 180))
                saturation = float(q.get("saturation", 0.25))
                contour = q.get("contours", "1") == "1"
                transparent = q.get("transparent", "0") == "1"
                settings = dict(
                    kind=kind, category=category, level=level, mode=mode,
                    lower=lower, upper=upper, threshold=threshold,
                    raw_visibility_threshold=lower + threshold * (upper - lower),
                    alpha=alpha, saturation=saturation, contours=contour,
                    transparent=transparent)
                result = render(image, field, coverage, kind, lower, upper,
                                threshold, alpha, saturation, contour, transparent)
                if url.path.startswith("/api/download/"):
                    np.savez_compressed(
                        buffer, field=field, coverage=coverage,
                        metadata_json=np.array(json.dumps(
                            dict(capture=state.metadata(index, record),
                                 display=settings), ensure_ascii=False)))
                    return self.send(
                        buffer.getvalue(), "application/octet-stream",
                        filename="val_{}_{}_field.npz".format(index, kind))
                result.save(buffer, format="PNG")
                self.send(buffer.getvalue(), "image/png",
                          filename="val_{}_{}.png".format(index, kind))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, IndexError) as error:
                self.json(dict(error=str(error)), 400)
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.json(dict(error="{}: {}".format(type(error).__name__, error)), 500)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def load_state(args):
    model, dataset = base.load_model_dataset(args)
    if base.model_family(model) != "OAG":
        raise ValueError("This teaser viewer supports only OAG models.")
    if getattr(model, "training_stage", None) == "rgb" or not (
            hasattr(model, "rgb_prompt_mapper") or hasattr(model, "_position_scores")):
        raise ValueError("This OAG config has no RGB-space acoustic position prior.")
    return TeaserState(model, dataset, args)


def main():
    from mmengine.config import DictAction
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--repo-root")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    args.alpha = 180
    if not 1 <= args.port <= 65535 or not 1 <= args.cache_size <= 16:
        parser.error("Invalid port or cache-size.")
    state = load_state(args)
    server = create_server(state, args.host, args.port)
    print("OAG teaser viewer: http://{}:{} ({} validation images)".format(
        args.host, server.server_port, len(state.images)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
