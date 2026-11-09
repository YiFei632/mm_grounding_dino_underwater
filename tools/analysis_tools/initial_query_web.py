"""Dependency-free validation query viewer, launched by visualize_initial_queries."""
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from mmengine.fileio import get


class ViewerState:
    def __init__(self, model, dataset, args, capture, project, capture_priors, overlay):
        self.model, self.dataset, self.args = model, dataset, args
        self.capture, self.project = capture, project
        self.capture_priors, self.overlay = capture_priors, overlay
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.images = []
        for index in range(len(dataset)):
            info = dataset.get_data_info(index)
            self.images.append(dict(index=index, name=Path(info['img_path']).name,
                                    img_id=str(info.get('img_id', ''))))

    def validate_index(self, index):
        if not 0 <= index < len(self.images):
            raise IndexError('Image index is outside the validation dataset.')

    def original(self, index):
        self.validate_index(index)
        path = self.dataset.get_data_info(index)['img_path']
        return Image.open(BytesIO(get(path))).convert('RGB')

    def queries(self, index):
        self.validate_index(index)
        # Capture temporarily replaces decoder.forward. Serialize all inference
        # and dataset transforms so requests cannot capture each other's data.
        with self.lock:
            if index in self.cache:
                self.cache.move_to_end(index)
                return self.cache[index]
            fields, meta = self.capture_priors(self.model, self.dataset[index])
            image = self.original(index)
            if image.size != tuple(reversed(meta['ori_shape'][:2])):
                raise ValueError('Original image dimensions differ from metadata.')
            record = dict(index=index, name=self.images[index]['name'],
                          original_shape=list(meta['ori_shape'][:2]),
                          normalization='Independent min/max per image and prior',
                          ranges={k: [float(v.min()), float(v.max())] for k, v in fields.items()},
                          fields=fields)
            self.cache[index] = record
            if len(self.cache) > 16:
                self.cache.popitem(last=False)
            return record


def create_server(state, host, port):
    html = Path(__file__).with_name('initial_query_viewer.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, content_type, status=200, filename=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value, status=200):
            self.send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                      'application/json; charset=utf-8', status)

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)
                if path == '/':
                    self.send(html, 'text/html; charset=utf-8')
                elif path == '/api/info':
                    self.json(dict(total=len(state.images), split='val',
                                   config=Path(state.args.config).name,
                                   checkpoint=Path(state.args.checkpoint).name,
                                   device=state.args.device, max_points=state.args.max_points,
                                   point_radius=state.args.point_radius,
                                   alpha=state.args.alpha, seed=state.args.seed))
                elif path == '/api/images':
                    term = query.get('q', [''])[0].casefold()
                    offset = max(0, int(query.get('offset', ['0'])[0]))
                    limit = min(100, max(1, int(query.get('limit', ['40'])[0])))
                    found = [i for i in state.images if term in i['name'].casefold()
                             or term == str(i['index']) or term == i['img_id'].casefold()]
                    self.json(dict(total=len(found), images=found[offset:offset+limit]))
                elif path.startswith(('/api/image/', '/api/queries/', '/api/download/', '/api/heatmap/')):
                    index = int(path.rsplit('/', 1)[1])
                    state.validate_index(index)
                    record = state.queries(index)
                    if path.startswith('/api/queries/'):
                        self.json({k: v for k, v in record.items() if k != 'fields'})
                    elif path.startswith('/api/download/'):
                        buffer = BytesIO()
                        np.savez_compressed(buffer, **record['fields'])
                        self.send(buffer.getvalue(), 'application/octet-stream',
                                  filename=f'val_{index:06d}_priors.npz')
                    else:
                        kind = query.get('kind', ['semantic'])[0]
                        if kind not in record['fields']:
                            raise ValueError('Unknown prior type')
                        alpha = int(query.get('alpha', [str(state.args.alpha)])[0])
                        if not 0 <= alpha <= 255:
                            raise ValueError('alpha must be in [0, 255]')
                        threshold = float(query.get('threshold', ['0'])[0])
                        if not 0 <= threshold <= 1:
                            raise ValueError('threshold must be in [0, 1]')
                        buffer = BytesIO()
                        state.overlay(state.original(index), record['fields'][kind], alpha, threshold).save(buffer, format='PNG')
                        self.send(buffer.getvalue(), 'image/png')
                else:
                    self.json(dict(error='Not found'), 404)
            except (ValueError, IndexError) as error:
                self.json(dict(error=str(error)), 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.json(dict(error=f'{type(error).__name__}: {error}'), 500)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_viewer(model, dataset, args, capture, project, capture_priors, overlay):
    state = ViewerState(model, dataset, args, capture, project, capture_priors, overlay)
    server = create_server(state, args.host, args.port)
    address = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    print(f'Validation query viewer: http://{address}:{server.server_port}\n'
          f'{len(state.images)} validation images; model loaded once. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
