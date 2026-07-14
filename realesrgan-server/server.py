"""
Unified Local AI Upscale Server
Supports: Real-ESRGAN + Swin2SR
Port: 5001
"""

import os
import sys
import base64
import io
import json
import logging
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
WEIGHTS_DIR  = BASE_DIR / 'weights'
SWIN2SR_DIR  = BASE_DIR / 'swin2sr-models'

# ── Model Registries ──────────────────────────────────────────────────────────

REALESRGAN_MODELS = {
    'realesrgan-x4plus': {
        'file':  'RealESRGAN_x4plus.pth',
        'scale': 4,
        'anime': False,
        'desc':  'General photo upscaling 4×',
    },
    'realesrgan-x4plus-anime': {
        'file':  'RealESRGAN_x4plus_anime_6B.pth',
        'scale': 4,
        'anime': True,
        'desc':  'Anime / illustration upscaling 4×',
    },
    'realesrgan-x2plus': {
        'file':  'RealESRGAN_x2plus.pth',
        'scale': 2,
        'anime': False,
        'desc':  'General photo upscaling 2×',
    },
    # ── High-quality community models ─────────────────────────────────────────
    '4x-UltraSharp': {
        'file':  '4x-UltraSharp.pth',
        'scale': 4,
        'anime': False,
        'desc':  '4× Ultra tajam — kualitas tertinggi, detail luar biasa',
    },
    '4x-Remacri': {
        'file':  '4x_foolhardy_Remacri.pth',
        'scale': 4,
        'anime': False,
        'desc':  '4× Remacri — detail halus & warna natural, mirip Clarity Pro',
    },
}

SWIN2SR_MODELS = {
    'swin2sr-realworld-x4': {
        'hf_id': 'caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr',
        'scale': 4,
        'desc':  'Real-world photo 4× — terbaik untuk foto/kamera',
    },
    'swin2sr-compressed-x4': {
        'hf_id': 'caidas/swin2SR-compressed-sr-x4-48',
        'scale': 4,
        'desc':  'Compressed JPEG 4× — optimal untuk gambar artefak JPEG',
    },
    'swin2sr-lightweight-x2': {
        'hf_id': 'caidas/swin2SR-lightweight-x2-64',
        'scale': 2,
        'desc':  'Lightweight 2× — cepat untuk gambar besar',
    },
    'swin2sr-classical-x4': {
        'hf_id': 'caidas/swin2SR-classical-sr-x4-64',
        'scale': 4,
        'desc':  'Classical SR 4× — ideal untuk gambar clean/illustration',
    },
}

# ── Model Caches ───────────────────────────────────────────────────────────────
_esrgan_cache = {}
_swin2sr_cache = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def b64_to_pil(b64_str: str) -> Image.Image:
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('RGB')

def pil_to_b64(img: Image.Image, fmt='PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

# ── Real-ESRGAN loader ────────────────────────────────────────────────────────
# Models that use 6 blocks (anime architecture)
_ANIME_MODELS  = {'realesrgan-x4plus-anime'}
# Models that use SPAN architecture (community models like Nomos8k)
_SPAN_MODELS   = {'4x-Nomos8k'}

def get_esrgan(model_id: str):
    if model_id in _esrgan_cache:
        return _esrgan_cache[model_id]

    cfg   = REALESRGAN_MODELS[model_id]
    wpath = WEIGHTS_DIR / cfg['file']
    if not wpath.exists():
        raise FileNotFoundError(
            f"Weight tidak ada: {wpath}\n"
            f"Jalankan: python realesrgan-server/download_weights.py"
        )

    from realesrgan import RealESRGANer

    if model_id in _SPAN_MODELS:
        # SPAN-based community model — use spandrel/net_interp fallback
        try:
            import spandrel
            model_obj = spandrel.ModelLoader().load_from_file(str(wpath)).model
        except Exception:
            # Fallback: load as generic ESRGAN
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model_obj = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=cfg['scale']
            )
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        num_block = 6 if model_id in _ANIME_MODELS else 23
        model_obj = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=num_block, num_grow_ch=32, scale=cfg['scale']
        )

    upsampler = RealESRGANer(
        scale=cfg['scale'], model_path=str(wpath), model=model_obj,
        tile=512, tile_pad=10, pre_pad=0, half=False, device=DEVICE,
    )
    _esrgan_cache[model_id] = upsampler
    log.info(f"✅ Real-ESRGAN loaded: {model_id} (scale={cfg['scale']}×, device={DEVICE})")
    return upsampler

# ── Swin2SR loader ────────────────────────────────────────────────────────────
def get_swin2sr(model_id: str):
    if model_id in _swin2sr_cache:
        return _swin2sr_cache[model_id]

    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    cfg     = SWIN2SR_MODELS[model_id]
    hf_id   = cfg['hf_id']
    loc_dir = SWIN2SR_DIR / model_id

    # Load from local cache if already downloaded, else fetch from HuggingFace
    src = str(loc_dir) if loc_dir.exists() else hf_id
    log.info(f"Loading Swin2SR: {model_id} from {src}")

    processor = Swin2SRImageProcessor.from_pretrained(src)
    model     = Swin2SRForImageSuperResolution.from_pretrained(src)
    model.eval()
    # MPS/CUDA if available
    if DEVICE.type in ('cuda', 'mps'):
        model = model.to(DEVICE)

    # Save locally for future offline use
    if not loc_dir.exists():
        log.info(f"Saving Swin2SR offline: {loc_dir}")
        loc_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(str(loc_dir))
        model.save_pretrained(str(loc_dir))

    _swin2sr_cache[model_id] = (processor, model)
    log.info(f"✅ Swin2SR loaded: {model_id} (scale={cfg['scale']}×, device={DEVICE})")
    return processor, model

def run_swin2sr(pil_in: Image.Image, model_id: str) -> Image.Image:
    processor, model = get_swin2sr(model_id)

    # Swin2SR works best on smaller tiles; split if needed
    w, h = pil_in.size
    MAX_SIDE = 256  # tile size for Swin2SR (model trained on small patches)

    if w <= MAX_SIDE and h <= MAX_SIDE:
        return _swin2sr_single(pil_in, processor, model, model_id)
    else:
        return _swin2sr_tiled(pil_in, processor, model, model_id, MAX_SIDE)

def _swin2sr_single(pil_in, processor, model, model_id):
    scale = SWIN2SR_MODELS[model_id]['scale']
    with torch.no_grad():
        inp = processor(images=pil_in, return_tensors='pt')
        if DEVICE.type in ('cuda', 'mps'):
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
        out = model(**inp)
    arr = out.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)  # CHW → HWC
    return Image.fromarray(arr)

def _swin2sr_tiled(pil_in, processor, model, model_id, tile_size):
    """Process large images tile by tile."""
    scale  = SWIN2SR_MODELS[model_id]['scale']
    w, h   = pil_in.size
    out_w, out_h = w * scale, h * scale
    out_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    pad = 8  # overlap to avoid seams

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w, x + tile_size + pad)
            y2 = min(h, y + tile_size + pad)

            tile   = pil_in.crop((x1, y1, x2, y2))
            tile_u = _swin2sr_single(tile, processor, model, model_id)
            tile_a = np.array(tile_u)

            # Determine paste region (trimming the overlap padding)
            px1 = (x - x1) * scale;  py1 = (y - y1) * scale
            px2 = px1 + min(tile_size, w - x) * scale
            py2 = py1 + min(tile_size, h - y) * scale

            ta = tile_a[py1:py2, px1:px2]
            ox1 = x * scale;  oy1 = y * scale
            ox2 = ox1 + ta.shape[1];  oy2 = oy1 + ta.shape[0]
            out_arr[oy1:oy2, ox1:ox2] = ta

    return Image.fromarray(out_arr)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})

@app.route('/models', methods=['GET'])
def list_models():
    result = []
    # Real-ESRGAN
    for mid, cfg in REALESRGAN_MODELS.items():
        result.append({
            'id':        mid,
            'engine':    'realesrgan',
            'desc':      cfg['desc'],
            'scale':     cfg['scale'],
            'available': (WEIGHTS_DIR / cfg['file']).exists(),
        })
    # Swin2SR
    for mid, cfg in SWIN2SR_MODELS.items():
        loc = SWIN2SR_DIR / mid
        result.append({
            'id':        mid,
            'engine':    'swin2sr',
            'desc':      cfg['desc'],
            'scale':     cfg['scale'],
            'available': loc.exists(),   # True after first download
        })
    return jsonify(result)

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'Field "image" (base64) wajib ada'}), 400

        model_id   = data.get('model', 'realesrgan-x4plus')
        engine     = data.get('engine', 'auto')  # 'realesrgan' | 'swin2sr' | 'auto'
        out_format = data.get('format', 'PNG').upper()

        # Auto-detect engine from model_id prefix
        if engine == 'auto':
            engine = 'swin2sr' if model_id.startswith('swin2sr') else 'realesrgan'

        pil_in = b64_to_pil(data['image'])
        log.info(f"Upscale: engine={engine}, model={model_id}, size={pil_in.size}")

        if engine == 'realesrgan':
            if model_id not in REALESRGAN_MODELS:
                return jsonify({'error': f'Model tidak dikenal: {model_id}'}), 400
            upsampler = get_esrgan(model_id)
            bgr_in    = np.array(pil_in)[:, :, ::-1]   # RGB→BGR
            bgr_out, _ = upsampler.enhance(bgr_in, outscale=REALESRGAN_MODELS[model_id]['scale'])
            pil_out   = Image.fromarray(bgr_out[:, :, ::-1].astype(np.uint8))

        elif engine == 'swin2sr':
            if model_id not in SWIN2SR_MODELS:
                return jsonify({'error': f'Model tidak dikenal: {model_id}'}), 400
            pil_out = run_swin2sr(pil_in, model_id)

        else:
            return jsonify({'error': f'Engine tidak dikenal: {engine}'}), 400

        log.info(f"✅ {pil_in.size} → {pil_out.size}")

        return jsonify({
            'image':  pil_to_b64(pil_out, out_format),
            'format': out_format,
            'width':  pil_out.width,
            'height': pil_out.height,
            'model':  model_id,
            'engine': engine,
        })

    except FileNotFoundError as e:
        log.error(str(e))
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ── Download Swin2SR models offline ───────────────────────────────────────────
@app.route('/download-swin2sr', methods=['POST'])
def download_swin2sr():
    """Trigger offline download of a Swin2SR model. Called once per model."""
    data     = request.get_json(force=True) or {}
    model_id = data.get('model', 'swin2sr-realworld-x4')
    if model_id not in SWIN2SR_MODELS:
        return jsonify({'error': f'Model tidak dikenal: {model_id}'}), 400
    try:
        get_swin2sr(model_id)   # load + auto-save locally
        return jsonify({'status': 'ok', 'model': model_id, 'saved': str(SWIN2SR_DIR / model_id)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    log.info(f"🚀 AI Upscale Server (Real-ESRGAN + Swin2SR) | device={DEVICE}")
    log.info(f"   Real-ESRGAN weights: {WEIGHTS_DIR}")
    log.info(f"   Swin2SR models:      {SWIN2SR_DIR}")
    esrgan_avail = [k for k, v in REALESRGAN_MODELS.items() if (WEIGHTS_DIR / v['file']).exists()]
    swin_avail   = [k for k in SWIN2SR_MODELS if (SWIN2SR_DIR / k).exists()]
    log.info(f"   Real-ESRGAN ready: {esrgan_avail}")
    log.info(f"   Swin2SR ready:     {swin_avail or '(akan diunduh saat pertama kali digunakan)'}")
    # threaded=True: /ping & /models tetap responsif saat /upscale sedang berjalan
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
