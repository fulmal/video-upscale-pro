/**
 * Real-ESRGAN Local Client
 * Berkomunikasi dengan server lokal di http://127.0.0.1:5001
 *
 * Cara pakai:
 *   1. Jalankan: ./realesrgan-server/start.sh
 *   2. Aktifkan "Real-ESRGAN (Offline)" di UI
 */

/** Base URL — proxied melalui Node.js Express server */
const REALESRGAN_BASE = '/api';

/** Cek apakah server Real-ESRGAN sedang berjalan */
async function realesrganPing() {
    try {
        const r = await fetch(`${REALESRGAN_BASE}/ping`, { signal: AbortSignal.timeout(5000) });
        if (r.ok) {
            const d = await r.json();
            return { ok: true, device: d.device };
        }
    } catch (_) {}
    return { ok: false };
}

/** Ambil daftar model yang tersedia dari server */
async function realesrganListModels() {
    const r = await fetch(`${REALESRGAN_BASE}/models`);
    return r.json(); // [{id, desc, scale, available}]
}

/**
 * Upscale gambar menggunakan Real-ESRGAN / Swin2SR offline.
 *
 * @param {string}   b64jpeg    - base64 JPEG data (tanpa prefix "data:...")
 * @param {string}   modelId    - 'realesrgan-x4plus' | 'swin2sr-realworld-x4' | dll
 * @param {function} onProgress - callback(msg, pct)
 * @returns {Promise<HTMLCanvasElement>}  - canvas hasil upscale
 */
async function realesrganUpscale(b64jpeg, modelId, onProgress) {
    if (!b64jpeg) throw new Error('Data gambar kosong.');

    onProgress('Mempersiapkan gambar untuk server lokal...', 10);
    addLog('img-log', `📤 Mengirim ke server lokal (${modelId})...`, 'ai');

    // Auto-detect engine from model ID
    const engine = modelId.startsWith('swin2sr') ? 'swin2sr' : 'realesrgan';
    const label  = engine === 'swin2sr' ? 'Swin2SR' : 'Real-ESRGAN';

    onProgress(`${label}: mengirim ke server lokal...`, 20);

    const t0   = performance.now();
    const resp = await fetch(`${REALESRGAN_BASE}/upscale`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image: b64jpeg, model: modelId, engine, format: 'PNG' }),
        signal:  AbortSignal.timeout(300000), // 5 menit timeout max
    });

    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
        addLog('img-log', `❌ ${label} error: ${err.error || resp.status}`, 'err');
        throw new Error(err.error || `${label} error ${resp.status}`);
    }

    const waitMsg = engine === 'swin2sr'
        ? `${label}: memproses... (pertama kali ~download model, selanjutnya cepat)`
        : `${label}: memproses... (10–60 detik tergantung ukuran)`;
    onProgress(waitMsg, 30);
    addLog('img-log', `⏳ ${label}: server sedang inference...`, 'ai');

    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    onProgress(`${label}: selesai, memuat hasil...`, 90);
    addLog('img-log', `📥 ${label}: menerima hasil (${elapsed}s), decode gambar...`, 'ai');

    // Decode base64 PNG hasil ke canvas
    const img = await new Promise((resolve, reject) => {
        const el = new Image();
        el.onload  = () => resolve(el);
        el.onerror = () => reject(new Error('Gagal decode gambar hasil dari server'));
        el.src = `data:image/png;base64,${data.image}`;
    });

    const outW = data.width  || img.naturalWidth;
    const outH = data.height || img.naturalHeight;

    if (!outW || !outH) throw new Error('Server mengembalikan gambar dengan dimensi 0.');

    const outCanvas = document.createElement('canvas');
    outCanvas.width  = outW;
    outCanvas.height = outH;
    outCanvas.getContext('2d').drawImage(img, 0, 0, outW, outH);

    addLog('img-log', `✨ ${label} selesai: ${outW}×${outH} (${elapsed}s total)`, 'ok');
    onProgress(`${label}: selesai ✓`, 96);
    return outCanvas;
}

