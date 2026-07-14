"""
Download Real-ESRGAN pre-trained weights.
Jalankan sekali sebelum start server.
"""
import urllib.request
import os
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

MODELS = {
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'RealESRGAN_x4plus_anime_6B.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
}

def download(name, url):
    path = WEIGHTS_DIR / name
    if path.exists():
        print(f'  ✓ {name} sudah ada ({path.stat().st_size // 1024 // 1024} MB)')
        return
    print(f'  ⬇ Mengunduh {name}...')
    tmp = str(path) + '.tmp'
    def progress(count, block, total):
        mb = count * block // 1024 // 1024
        tot = total // 1024 // 1024
        print(f'\r    {mb}/{tot} MB', end='', flush=True)
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    os.rename(tmp, path)
    print(f'\n  ✅ {name} selesai ({path.stat().st_size // 1024 // 1024} MB)')

if __name__ == '__main__':
    print('📦 Download Real-ESRGAN weights...\n')
    for name, url in MODELS.items():
        download(name, url)
    print('\n✅ Semua weight siap! Jalankan: ./start.sh')
