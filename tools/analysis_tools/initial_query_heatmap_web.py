#!/usr/bin/env python
"""Interactive initial-query density viewer for OAG and Sonarbind-Prism."""
import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


def density_field(pixels, visible, shape, sigma=0.025):
    """Gaussian query probability density on a common normalized 256² grid.

    All visible queries have equal mass. Sigma is a fraction of image width /
    height, so models and images share a meaningful, resolution-free scale.
    """
    if not np.isfinite(sigma) or not 0.002 <= sigma <= 0.2:
        raise ValueError('sigma must be in [0.002, 0.2]')
    h, w = shape[:2]
    points = np.asarray(pixels)[np.asarray(visible, dtype=bool)] / [w, h]
    grid = (np.arange(256, dtype=np.float64) + 0.5) / 256
    field = np.zeros((256, 256), dtype=np.float64)
    for x, y in points:
        field += np.exp(-0.5*((grid-x)/sigma)**2)[None, :] * np.exp(-0.5*((grid-y)/sigma)**2)[:, None]
    if len(points):
        field /= len(points) * 2 * np.pi * sigma**2
    return field.astype(np.float32)


def overlay(image, field, alpha, lower, upper):
    if not all(np.isfinite(x) for x in (lower, upper)) or not 0 <= lower < upper:
        raise ValueError('Expected 0 <= lower < upper')
    if not 0 <= alpha <= 255:
        raise ValueError('alpha must be in [0, 255]')
    values = np.asarray(Image.fromarray(field).resize(image.size, Image.Resampling.BILINEAR))
    t = np.clip(values / upper, 0, 1)
    rgb = np.stack([np.clip(1.5-abs(4*t-3), 0, 1), np.clip(1.5-abs(4*t-2), 0, 1), np.clip(1.5-abs(4*t-1), 0, 1)], -1)
    mask = Image.fromarray(np.where((values >= lower) & (values > 0), alpha, 0).astype('uint8'))
    return Image.composite(Image.fromarray((rgb*255).astype('uint8')), image.convert('RGB'), mask)


def create_server(model, dataset, args, capture, project):
    from mmengine.fileio import get
    lock, cache = threading.Lock(), OrderedDict()
    images = [dict(index=i, name=Path(dataset.get_data_info(i)['img_path']).name,
                   img_id=str(dataset.get_data_info(i).get('img_id', ''))) for i in range(len(dataset))]

    def record(index):
        if not 0 <= index < len(images):
            raise ValueError('Image index outside dataset')
        with lock:
            if index not in cache:
                refs, meta = capture(model, dataset[index])
                if len(refs) != model.num_queries:
                    raise ValueError('Expected exactly model.num_queries initial references')
                pixels, visible = project(refs, meta)
                cache[index] = dict(references_normalized=refs, centers_original_xy=pixels,
                                    visible=visible, original_shape=np.array(meta['ori_shape'][:2]))
                if len(cache) > 16:
                    cache.popitem(last=False)
            cache.move_to_end(index)
            return cache[index]

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, kind, code=200, filename=None):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value, code=200):
            self.send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', code)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == '/':
                    return self.send(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
                if url.path == '/api/info':
                    return self.json(dict(total=len(images), config=args.config, checkpoint=args.checkpoint,
                                          model=type(model).__name__, sigma=args.sigma, upper=args.upper, lower=args.lower, alpha=args.alpha))
                if url.path == '/api/images':
                    term = q.get('q', '').casefold()
                    found = [v for v in images if term in v['name'].casefold() or term in (str(v['index']), v['img_id'].casefold())]
                    offset = max(0, int(q.get('offset', 0)))
                    return self.json(dict(total=len(found), images=found[offset:offset+40]))
                if not url.path.startswith(('/api/queries/', '/api/heatmap/', '/api/download/', '/api/original/')):
                    return self.json(dict(error='Not found'), 404)
                index = int(url.path.rsplit('/', 1)[1])
                data = record(index)
                if url.path.startswith('/api/queries/'):
                    return self.json(dict(name=images[index]['name'], total=len(data['visible']), visible=int(data['visible'].sum())))
                sigma = float(q.get('sigma', args.sigma))
                field = density_field(data['centers_original_xy'], data['visible'], data['original_shape'], sigma)
                buffer = BytesIO()
                if url.path.startswith('/api/download/'):
                    np.savez_compressed(buffer, **data, density=field, sigma=sigma)
                    return self.send(buffer.getvalue(), 'application/octet-stream', filename=f'queries_{index}.npz')
                image = Image.open(BytesIO(get(dataset.get_data_info(index)['img_path']))).convert('RGB')
                if image.size != tuple(reversed(data['original_shape'])):
                    raise ValueError('Original image dimensions differ from metadata')
                if not url.path.startswith('/api/original/'):
                    image = overlay(image, field, int(q.get('alpha', args.alpha)), float(q.get('lower', args.lower)), float(q.get('upper', args.upper)))
                image.save(buffer, format='PNG')
                self.send(buffer.getvalue(), 'image/png', filename=f'queries_{index}.png')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, IndexError) as error:
                self.json(dict(error=str(error)), 400)
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.json(dict(error=f'{type(error).__name__}: {error}'), 500)

    return ThreadingHTTPServer((args.host, args.port), Handler)


def main():
    from visualize_initial_queries import (Config, DictAction, DATASETS, MODELS, register_all_modules,
        init_default_scope, import_modules_from_strings, load_checkpoint, capture_initial_queries, centers_in_original_image)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--sigma', type=float, default=0.025)
    parser.add_argument('--lower', type=float, default=0.)
    parser.add_argument('--upper', type=float, default=10.)
    parser.add_argument('--alpha', type=int, default=180)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if not (0.002 <= args.sigma <= 0.2 and 0 <= args.lower < args.upper < float('inf') and 0 <= args.alpha <= 255 and 1 <= args.port <= 65535):
        parser.error('Invalid sigma, lower/upper, alpha or port')
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.model.get('test_cfg', {}).get('chunked_size', -1) > 0:
        raise ValueError('Set model.test_cfg.chunked_size=-1; chunked prompts are unsupported')
    register_all_modules(init_default_scope=False)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.to(args.device).eval()
    server = create_server(model, dataset, args, capture_initial_queries, centers_in_original_image)
    print(f'Initial query heatmap: http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
