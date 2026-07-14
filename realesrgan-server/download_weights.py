"""
Download Real-ESRGAN pre-trained weights.
Jalankan sekali sebelum start server.

Tersedia:
  - RealESRGAN_x4plus.pth        (foto umum 4×)
  - RealESRGAN_x4plus_anime_6B   (anime/ilustrasi 4×)
  - RealESRGAN_x2plus.pth        (foto umum 2×)
  - 4x-UltraSharp.pth            (kualitas ultra tajam ⭐⭐⭐⭐⭐)
  - 4x_Remacri.pth               (detail halus, mirip Clarity Pro ⭐⭐⭐⭐⭐)
  - 4x_Nomos8k_span_otf_fast.pth (texture tinggi ⭐⭐⭐⭐)
"""
import urllib.request
import sys
import os
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

# ── Model list ────────────────────────────────────────────────────────────────
# Official Real-ESRGAN models
OFFICIAL_MODELS = {
    'RealESRGAN_x4plus.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    ),
    'RealESRGAN_x4plus_anime_6B.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    ),
    'RealESRGAN_x2plus.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    ),
}

# High-quality community models (dari OpenModelDB / HuggingFace)
COMMUNITY_MODELS = {
    '4x-UltraSharp.pth': (
        'https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth'
    ),
    '4x_foolhardy_Remacri.pth': (
        'https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth'
    ),
}

ALL_MODELS = {**OFFICIAL_MODELS, **COMMUNITY_MODELS}

# ── Download helper ───────────────────────────────────────────────────────────
def download(name, url):
    path = WEIGHTS_DIR / name
    if path.exists():
        size_mb = path.stat().st_size // 1024 // 1024
        print(f'  ✓ {name} sudah ada ({size_mb} MB)')
        return True

    print(f'  ⬇ Mengunduh {name}...')
    print(f'    URL: {url}')
    tmp = str(path) + '.tmp'

    try:
        def progress(count, block, total):
            if total > 0:
                pct = min(count * block / total * 100, 100)
                mb  = count * block // 1024 // 1024
                tot = total // 1024 // 1024
                print(f'\r    {mb}/{tot} MB ({pct:.0f}%)', end='', flush=True)
            else:
                mb = count * block // 1024 // 1024
                print(f'\r    {mb} MB', end='', flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        os.rename(tmp, path)
        size_mb = path.stat().st_size // 1024 // 1024
        print(f'\n  ✅ {name} selesai ({size_mb} MB)')
        return True

    except Exception as e:
        print(f'\n  ❌ Gagal download {name}: {e}')
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Pilih model yang akan didownload
    # Usage: python download_weights.py [all | official | community | <nama_file>]
    arg = sys.argv[1] if len(sys.argv) > 1 else 'official'

    if arg == 'all':
        targets = ALL_MODELS
        print('📦 Download SEMUA model weights...\n')
    elif arg == 'community':
        targets = COMMUNITY_MODELS
        print('📦 Download Community models (UltraSharp, Remacri, Nomos8k)...\n')
    elif arg == 'official':
        targets = OFFICIAL_MODELS
        print('📦 Download Official Real-ESRGAN weights...\n')
    elif arg in ALL_MODELS:
        targets = {arg: ALL_MODELS[arg]}
        print(f'📦 Download {arg}...\n')
    else:
        print(f'❌ Model tidak dikenal: {arg}')
        print(f'   Tersedia: {list(ALL_MODELS.keys())}')
        sys.exit(1)

    ok = 0
    fail = 0
    for name, url in targets.items():
        if download(name, url):
            ok += 1
        else:
            fail += 1

    print(f'\n{"✅" if fail == 0 else "⚠️"} Selesai: {ok} berhasil, {fail} gagal')
    if ok > 0:
        print('   Jalankan server: ./realesrgan-server/start.sh')
