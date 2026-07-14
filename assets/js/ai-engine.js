/**
 * ScalerPro — AI Generate Upscale Engine
 * Supports: HuggingFace Inference API (free) + Replicate API (paid) + Google Gemini
 * 100% client-side — API keys stored in localStorage, never sent anywhere else.
 */

// ═══════════════════════════════════════════════════════
// Default API Key (pre-configured)
// ═══════════════════════════════════════════════════════
const DEFAULT_GEMINI_KEY = '';
// Version stamp — change this whenever DEFAULT_GEMINI_KEY changes to auto-clear old cache
const GEMINI_KEY_VERSION  = 'v3';

// ═══════════════════════════════════════════════════════
// Model Registry
// ═══════════════════════════════════════════════════════
const AI_MODELS = {
    huggingface: [
        {
            id:    'swin2sr_realworld',
            name:  'Swin2SR — Real-World Photo',
            url:   'https://api-inference.huggingface.co/models/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr',
            scale: 4,
            desc:  'Terbaik untuk foto nyata (JPEG/kamera)',
        },
        {
            id:    'swin2sr_compressed',
            name:  'Swin2SR — Compressed Image',
            url:   'https://api-inference.huggingface.co/models/caidas/swin2SR-compressed-sr-x4-48',
            scale: 4,
            desc:  'Cocok untuk gambar hasil kompresi/artefak JPEG',
        },
        {
            id:    'swin2sr_lightweight',
            name:  'Swin2SR — Lightweight (cepat)',
            url:   'https://api-inference.huggingface.co/models/caidas/swin2SR-lightweight-sr-x2-64',
            scale: 2,
            desc:  'Lebih cepat, 2× upscale, cocok untuk gambar besar',
        },
    ],
    replicate: [
        {
            id:      'real_esrgan',
            name:    'Real-ESRGAN 4×+',
            version: '42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b',
            scale:   4,
            desc:    'Kualitas tertinggi, detail foto sangat realistis',
        },
        {
            id:      'real_esrgan_face',
            name:    'Real-ESRGAN + Face Restore',
            version: '42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b',
            scale:   4,
            faceEnhance: true,
            desc:    'Khusus foto dengan wajah — perbaiki detail wajah',
        },
        {
            id:      'swinir',
            name:    'SwinIR — Image Restoration',
            version: '660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9ce60e16f3ae1',
            scale:   4,
            desc:    'Denoise + Super-Resolution berbasis Swin Transformer',
        },
    ],
    // ── Local AI Server (Real-ESRGAN + Swin2SR) — port 5001 ──
    realesrgan_local: [
        // ── Real-ESRGAN ──
        {
            id:     'realesrgan-x4plus',
            name:   '📷 Real-ESRGAN 4× — Photo',
            engine: 'realesrgan',
            scale:  4,
            desc:   'Real-ESRGAN — kualitas terbaik untuk foto nyata, detail sangat tajam',
        },
        {
            id:     'realesrgan-x4plus-anime',
            name:   '🎨 Real-ESRGAN 4× — Anime',
            engine: 'realesrgan',
            scale:  4,
            desc:   'Real-ESRGAN — dioptimalkan untuk ilustrasi, anime, kartun',
        },
        {
            id:     'realesrgan-x2plus',
            name:   '⚡ Real-ESRGAN 2× — Fast',
            engine: 'realesrgan',
            scale:  2,
            desc:   'Real-ESRGAN — cepat, cocok untuk gambar yang sudah besar',
        },
        // ── Swin2SR ──
        {
            id:     'swin2sr-realworld-x4',
            name:   '🌄 Swin2SR 4× — Real World',
            engine: 'swin2sr',
            scale:  4,
            desc:   'Swin2SR — Transformer-based, terbaik untuk foto nyata & kamera',
        },
        {
            id:     'swin2sr-compressed-x4',
            name:   '🖼 Swin2SR 4× — JPEG Fix',
            engine: 'swin2sr',
            scale:  4,
            desc:   'Swin2SR — optimal untuk gambar terkompresi & artefak JPEG',
        },
        {
            id:     'swin2sr-classical-x4',
            name:   '✦ Swin2SR 4× — Classical',
            engine: 'swin2sr',
            scale:  4,
            desc:   'Swin2SR — classical SR, ideal untuk ilustrasi & gambar clean',
        },
        {
            id:     'swin2sr-lightweight-x2',
            name:   '🚀 Swin2SR 2× — Lightweight',
            engine: 'swin2sr',
            scale:  2,
            desc:   'Swin2SR — paling cepat & ringan, cocok untuk gambar HD',
        },
    ],
    gemini: [
        {
            id:    'nano_banana_pro',
            name:  '🍌 Nano Banana Pro',
            model: 'gemini-2.0-flash-lite',
            desc:  'Paling ringan & cepat — Gemini 2.0 Flash Lite, optimal untuk analisa gambar',
        },
        {
            id:    'flash25',
            name:  'Gemini 2.5 Flash',
            model: 'gemini-2.5-flash',
            desc:  'Terbaru & cerdas — Gemini 2.5 Flash untuk analisa mendalam',
        },
        {
            id:    'flash20',
            name:  'Gemini 2.0 Flash',
            model: 'gemini-2.0-flash',
            desc:  'Stabil & reliable — Gemini 2.0 Flash standar',
        },
    ],
};

// ═══════════════════════════════════════════════════════
// LocalStorage Key Management
// ═══════════════════════════════════════════════════════
const LS_KEY_HF  = 'scalerpro_hf_key';
const LS_KEY_REP = 'scalerpro_rep_key';
const LS_KEY_GEM = 'scalerpro_gem_key';

function saveKey(provider, key) {
    const k = provider === 'huggingface' ? LS_KEY_HF
            : provider === 'replicate'   ? LS_KEY_REP
            :                              LS_KEY_GEM;
    try { localStorage.setItem(k, key); } catch(_) {}
}
function loadKey(provider) {
    const k = provider === 'huggingface' ? LS_KEY_HF
            : provider === 'replicate'   ? LS_KEY_REP
            :                              LS_KEY_GEM;
    try {
        return localStorage.getItem(k) || (provider === 'gemini' ? DEFAULT_GEMINI_KEY : '');
    } catch(_) {
        return provider === 'gemini' ? DEFAULT_GEMINI_KEY : '';
    }
}
function clearKey(provider) {
    const k = provider === 'huggingface' ? LS_KEY_HF
            : provider === 'replicate'   ? LS_KEY_REP
            :                              LS_KEY_GEM;
    try { localStorage.removeItem(k); } catch(_) {}
}

// ═══════════════════════════════════════════════════════
// Utility Helpers
// ═══════════════════════════════════════════════════════
function canvasToBlob(canvas, type = 'image/png', quality = 0.95) {
    return new Promise(res => canvas.toBlob(res, type, quality));
}

function blobToBase64(blob) {
    return new Promise((res, rej) => {
        const r = new FileReader();
        r.onload  = () => res(r.result);
        r.onerror = rej;
        r.readAsDataURL(blob);
    });
}

function loadImgFromBlob(blob) {
    return new Promise((res, rej) => {
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload  = () => { URL.revokeObjectURL(url); res(img); };
        img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('Gagal memuat gambar AI')); };
        img.src = url;
    });
}

/** Compress canvas to ≤ maxBytes (approximate) for API size limits */
async function compressForAPI(canvas, maxBytes = 8 * 1024 * 1024) {
    // Try PNG first
    let blob = await canvasToBlob(canvas, 'image/png');
    if (blob.size <= maxBytes) return { blob, w: canvas.width, h: canvas.height };

    // Resize down to fit
    const ratio  = Math.sqrt(maxBytes / blob.size);
    const newW   = Math.floor(canvas.width  * ratio);
    const newH   = Math.floor(canvas.height * ratio);
    const tmp    = document.createElement('canvas');
    tmp.width    = newW; tmp.height = newH;
    tmp.getContext('2d').drawImage(canvas, 0, 0, newW, newH);

    blob = await canvasToBlob(tmp, 'image/jpeg', 0.88); // JPEG for size
    return { blob, w: newW, h: newH };
}

/** Draw AI result onto a canvas of the ORIGINAL requested output size */
function buildFinalCanvas(aiImg, targetW, targetH, webglCanvas, strength) {
    const out = document.createElement('canvas');
    out.width  = targetW;
    out.height = targetH;
    const ctx  = out.getContext('2d');

    if (strength < 1.0) {
        // Underlay: WebGL result at target size
        ctx.drawImage(webglCanvas, 0, 0, targetW, targetH);
        ctx.globalAlpha = strength;
    }
    // Overlay: AI result scaled to target size
    ctx.drawImage(aiImg, 0, 0, targetW, targetH);
    ctx.globalAlpha = 1.0;
    return out;
}

// ═══════════════════════════════════════════════════════
// HuggingFace Inference API
// ═══════════════════════════════════════════════════════
async function hfUpscale(webglCanvas, targetW, targetH, opts) {
    const { apiKey, modelId, strength, onProgress } = opts;
    const model = AI_MODELS.huggingface.find(m => m.id === modelId) || AI_MODELS.huggingface[0];

    onProgress('Menyiapkan gambar untuk AI...', 5);
    const { blob } = await compressForAPI(webglCanvas);

    onProgress('Menghubungi HuggingFace API...', 15);
    let resp = await fetchHF(model.url, apiKey, blob);

    // 503 = model sedang cold-start, tunggu dan coba lagi
    if (resp.status === 503) {
        const meta = await resp.json().catch(() => ({}));
        const wait = Math.min((meta.estimated_time || 25) * 1000, 60_000);
        for (let elapsed = 0; elapsed < wait; elapsed += 3000) {
            onProgress(`Model AI loading... ${Math.round((elapsed / wait) * 100)}%`, 15 + (elapsed / wait) * 25);
            await aiSleep(3000);
        }
        onProgress('Mencoba ulang...', 40);
        resp = await fetchHF(model.url, apiKey, blob);
    }

    if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(err?.error || `HuggingFace error ${resp.status}: ${resp.statusText}`);
    }

    onProgress('Menerima hasil dari AI...', 70);
    const resultBlob = await resp.blob();
    if (!resultBlob || resultBlob.size < 200) throw new Error('AI mengembalikan respons kosong. Coba lagi.');

    onProgress('Merender hasil AI...', 88);
    const aiImg = await loadImgFromBlob(resultBlob);

    onProgress('Menggabungkan AI + WebGL...', 96);
    const out = buildFinalCanvas(aiImg, targetW, targetH, webglCanvas, strength);
    return out;
}

async function fetchHF(url, apiKey, blob) {
    return fetch(url, {
        method:  'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type':  blob.type || 'application/octet-stream',
            'x-wait-for-model': 'true',
        },
        body: blob,
    });
}

// ═══════════════════════════════════════════════════════
// Replicate API
// ═══════════════════════════════════════════════════════
async function replicateUpscale(webglCanvas, targetW, targetH, opts) {
    const { apiKey, modelId, strength, onProgress } = opts;
    const model = AI_MODELS.replicate.find(m => m.id === modelId) || AI_MODELS.replicate[0];

    onProgress('Mengonversi gambar...', 5);
    const { blob } = await compressForAPI(webglCanvas);
    const base64   = await blobToBase64(blob);

    onProgress('Mengirim ke Replicate API...', 15);
    const createResp = await fetch('https://api.replicate.com/v1/predictions', {
        method:  'POST',
        headers: {
            'Authorization':  `Token ${apiKey}`,
            'Content-Type':   'application/json',
        },
        body: JSON.stringify({
            version: model.version,
            input: {
                image:        base64,
                scale:        2,
                face_enhance: model.faceEnhance || false,
            },
        }),
    });

    if (!createResp.ok) {
        const err = await createResp.json().catch(() => null);
        throw new Error(err?.detail || `Replicate error ${createResp.status}`);
    }

    const prediction = await createResp.json();
    const pollUrl    = prediction.urls?.get || `https://api.replicate.com/v1/predictions/${prediction.id}`;

    // Poll until done (max 3 min)
    onProgress('AI sedang generate detail...', 25);
    for (let i = 0; i < 90; i++) {
        await aiSleep(2000);
        const pollResp = await fetch(pollUrl, { headers: { 'Authorization': `Token ${apiKey}` } });
        if (!pollResp.ok) throw new Error(`Poll error: ${pollResp.status}`);
        const result = await pollResp.json();

        if (result.status === 'succeeded') {
            const pct = 25 + Math.min(i / 90, 1) * 55;
            onProgress('Mengunduh hasil AI...', pct);

            const dlResp    = await fetch(result.output);
            const dlBlob    = await dlResp.blob();
            const aiImg     = await loadImgFromBlob(dlBlob);

            onProgress('Menggabungkan AI + WebGL...', 96);
            return buildFinalCanvas(aiImg, targetW, targetH, webglCanvas, strength);
        }

        if (result.status === 'failed' || result.status === 'canceled') {
            throw new Error(result.error || `AI processing ${result.status}`);
        }

        const pct = 25 + Math.min(i / 90, 1) * 55;
        onProgress(`AI generate detail... ${Math.round(pct)}%`, pct);
    }
    throw new Error('Timeout — AI melebihi 3 menit. Coba gambar lebih kecil.');
}

// ═══════════════════════════════════════════════════════
// Google Gemini API
// ═══════════════════════════════════════════════════════

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

/** Helper: canvas → compact JPEG base64 string (strips data: prefix) */
async function canvasToJpegB64(canvas, quality = 0.88, maxPx = 1536) {
    let src = canvas;
    // Downsample if too large for Gemini (keeps under ~5MB)
    if (canvas.width > maxPx || canvas.height > maxPx) {
        const scale = Math.min(maxPx / canvas.width, maxPx / canvas.height);
        const tmp   = document.createElement('canvas');
        tmp.width   = Math.floor(canvas.width  * scale);
        tmp.height  = Math.floor(canvas.height * scale);
        tmp.getContext('2d').drawImage(canvas, 0, 0, tmp.width, tmp.height);
        src = tmp;
    }
    const blob   = await canvasToBlob(src, 'image/jpeg', quality);
    const b64url = await blobToBase64(blob);
    return b64url.split(',')[1]; // strip "data:image/jpeg;base64,"
}

/**
 * Gemini AI — Auto Settings for image upscale.
 * Analyzes the image → gets optimal enhancement params → applies via WebGL.
 * Gemini does NOT generate or modify pixels directly.
 */
async function geminiImgUpscale(webglCanvas, targetW, targetH, opts) {
    const { apiKey, modelId, onProgress } = opts;
    const modelInfo = AI_MODELS.gemini.find(m => m.id === modelId) || AI_MODELS.gemini[0];

    // Fallback chain: try selected model first, then reliable fallbacks
    const tryModels = [
        modelInfo.model,
        'gemini-2.5-flash',
        'gemini-2.0-flash-lite',
        'gemini-2.0-flash',
        'gemini-1.5-flash',
    ].filter((v, i, a) => a.indexOf(v) === i); // dedupe

    // ── Analyze: send image to Gemini → get optimal enhancement params ──
    onProgress(`${modelInfo.name} menganalisa gambar...`, 15);
    const b64 = await canvasToJpegB64(webglCanvas, 0.85, 1280);

    const reqBody = JSON.stringify({
        contents: [{
            parts: [
                {
                    text: `Analyze this image and return optimal GPU enhancement parameters for upscaling.

Evaluate the image carefully:
1. noise: grain/compression artifacts level (0=clean, 1=very noisy)
2. sharp: how much sharpening needed (0=none, 1=maximum)
3. clar: local contrast/clarity boost needed (0=none, 1=maximum)
4. vib: color vibrance boost (0=none, 1=maximum)
5. dens: detail synthesis density (0=minimal, 1=maximum)
6. sceneType: outdoor / indoor / portrait / nighttime / animation / document / other

Return ONLY valid JSON, no explanation:
{"sharp":<0-1>,"clar":<0-1>,"vib":<0-1>,"dens":<0-1>,"noise":<0-1>,"sceneType":"<type>","reasoning":"<one line>"}`
                },
                { inline_data: { mime_type: 'image/jpeg', data: b64 } }
            ]
        }],
        generationConfig: { responseMimeType: 'application/json' }
    });

    // Try each model until one works
    let aiParams = null;
    for(const model of tryModels) {
        try {
            onProgress(`${modelInfo.name} menganalisa (${model})...`, 15);
            const resp = await fetch(
                `${GEMINI_BASE}/${model}:generateContent?key=${apiKey}`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: reqBody }
            );

            if(!resp.ok) {
                const err = await resp.json().catch(() => null);
                const msg = err?.error?.message || `HTTP ${resp.status}`;
                if(resp.status === 404 || resp.status === 400) { console.warn(`[Gemini] ${model}: ${msg}`); continue; }
                if(resp.status === 403 || resp.status === 401) throw new Error('API key tidak valid.');
                continue;
            }

            const data    = await resp.json();
            const rawText = data.candidates?.[0]?.content?.parts?.[0]?.text || '{}';
            const parsed  = JSON.parse(rawText.replace(/```json|```/g, '').trim());
            aiParams = {
                sharp: Math.min(1, Math.max(0, +parsed.sharp ?? 0.5)),
                clar:  Math.min(1, Math.max(0, +parsed.clar  ?? 0.4)),
                vib:   Math.min(1, Math.max(0, +parsed.vib   ?? 0.3)),
                dens:  Math.min(1, Math.max(0, +parsed.dens  ?? 0.5)),
                noise: Math.min(1, Math.max(0, +parsed.noise ?? 0.2)),
                sceneType: parsed.sceneType || 'other',
            };
            console.log(`[Gemini ✅ ${model}] scene="${aiParams.sceneType}"`, aiParams, '→', parsed.reasoning);
            break; // success!
        } catch(e) {
            if(e.message.includes('API key') || e.message.includes('tidak valid')) throw e;
            console.warn(`[Gemini] ${model} gagal:`, e.message);
        }
    }

    if(!aiParams) {
        console.warn('[Gemini] Semua model gagal — pakai parameter default');
        aiParams = { sharp: 0.5, clar: 0.4, vib: 0.3, dens: 0.5, noise: 0.2, sceneType: 'other' };
    }

    // ── Apply AI params: render WebGL with optimized settings ──
    onProgress('Menerapkan Auto Settings ke WebGL...', 60);

    // Create fresh 2D output canvas (never mix WebGL + 2D on same canvas)
    const outputCanvas = document.createElement('canvas');
    outputCanvas.width  = targetW;
    outputCanvas.height = targetH;

    if(aiParams && typeof initImgGL === 'function' && typeof renderImgGL === 'function') {
        try {
            const glCanvas = document.createElement('canvas');
            glCanvas.width  = targetW;
            glCanvas.height = targetH;
            const glRef = initImgGL(glCanvas);
            if(glRef) {
                renderImgGL(glRef, webglCanvas, webglCanvas.width, webglCanvas.height, aiParams);
                glRef.gl.finish();
                outputCanvas.getContext('2d').drawImage(glCanvas, 0, 0, targetW, targetH);
            } else {
                outputCanvas.getContext('2d').drawImage(webglCanvas, 0, 0, targetW, targetH);
            }
        } catch(e) {
            console.warn('[Gemini] WebGL render gagal, fallback 2D scale:', e);
            outputCanvas.getContext('2d').drawImage(webglCanvas, 0, 0, targetW, targetH);
        }
    } else {
        outputCanvas.getContext('2d').drawImage(webglCanvas, 0, 0, targetW, targetH);
    }

    onProgress('Auto Settings selesai ✓', 96);
    return outputCanvas;
}


/**
 * VIDEO frame analysis via Gemini (vision/text model).
 * Sends one keyframe, returns optimal WebGL shader params.
 * Does NOT process video frames — only advises on settings.
 *
 * @param {HTMLCanvasElement} frameCanvas  — a video frame captured by app.js
 * @param {string}            apiKey       — Gemini API key
 * @returns {Promise<{sharp, clar, vib, dens, noise, sceneType}>}
 */
async function geminiVideoAnalyze(frameCanvas, apiKey) {
    const b64 = await canvasToJpegB64(frameCanvas, 0.85, 1024);

    const resp = await fetch(
        `${GEMINI_BASE}/gemini-2.0-flash:generateContent?key=${apiKey}`,
        {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{
                    parts: [
                        {
                            text: `Analyze this video frame and return a JSON object with optimal video enhancement settings.

Analyze:
1. Noise level (grain, compression artifacts)
2. Sharpness of edges and details
3. Color saturation and vibrancy
4. Overall image quality
5. Scene type (outdoor, indoor, face/portrait, nighttime, animation, text/document)

Return ONLY valid JSON (no markdown, no explanation), with these fields:
{
  "sharp": <0.0-1.0, how much sharpening/unsharp-mask>,
  "clar": <0.0-1.0, clarity/local contrast>,
  "vib": <0.0-1.0, color vibrance boost>,
  "dens": <0.0-2.0, pixel density synthesis — use 1.0 for normal, >1.0 for soft video>,
  "noise": <0.0-1.0, denoise strength — higher for grainy/noisy video>,
  "sceneType": "outdoor|indoor|portrait|nighttime|animation|document|other",
  "reasoning": "<one sentence why>"
}

Provide values that will make the video look its absolute best when upscaled.`
                        },
                        {
                            inline_data: { mime_type: 'image/jpeg', data: b64 }
                        }
                    ]
                }],
                generationConfig: {
                    responseMimeType: 'application/json',
                }
            })
        }
    );

    if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(err?.error?.message || `Gemini analyze error ${resp.status}`);
    }

    const data    = await resp.json();
    const rawText = data.candidates?.[0]?.content?.parts?.[0]?.text || '{}';

    // Parse and clamp values
    const p = JSON.parse(rawText.replace(/```json|```/g, '').trim());
    return {
        sharp:     Math.min(Math.max(+p.sharp     || 0.6, 0), 1.0),
        clar:      Math.min(Math.max(+p.clar      || 0.5, 0), 1.0),
        vib:       Math.min(Math.max(+p.vib       || 0.3, 0), 1.0),
        dens:      Math.min(Math.max(+p.dens      || 1.0, 0), 2.0),
        noise:     Math.min(Math.max(+p.noise     || 0.2, 0), 1.0),
        sceneType: p.sceneType || 'other',
        reasoning: p.reasoning || '',
    };
}

// ═══════════════════════════════════════════════════════
// Public Entry Point
// ═══════════════════════════════════════════════════════
/**
 * @param {HTMLCanvasElement} webglCanvas  — output dari WebGL pipeline
 * @param {number}            targetW      — lebar output akhir yang diinginkan
 * @param {number}            targetH      — tinggi output akhir yang diinginkan
 * @param {object}            opts
 *   .provider   'huggingface' | 'replicate'
 *   .apiKey     string
 *   .modelId    string
 *   .strength   0.0–1.0  (1.0 = 100% AI)
 *   .onProgress (msg, pct) => void
 * @returns {Promise<HTMLCanvasElement>}
 */
async function aiUpscale(webglCanvas, targetW, targetH, opts) {
    const provider = opts.provider || 'huggingface';

    // Real-ESRGAN / Swin2SR local — no API key needed
    if (provider === 'realesrgan_local') {
        if (typeof realesrganUpscale !== 'function') {
            throw new Error('realesrgan-client.js belum dimuat. Periksa index.html.');
        }

        // Terima base64 data URL string ATAU canvas element
        // (processImg mengirim string untuk menghindari volatile WebGL buffer)
        let b64;
        if (typeof webglCanvas === 'string') {
            // Sudah berupa data URL dari processImg
            b64 = webglCanvas.includes(',') ? webglCanvas.split(',')[1] : webglCanvas;
        } else {
            // Canvas element — ambil data URL sinkron sekarang
            try {
                const dataUrl = webglCanvas.toDataURL('image/jpeg', 0.92);
                b64 = dataUrl.split(',')[1];
            } catch (e) {
                throw new Error('Gagal membaca gambar dari canvas: ' + e.message);
            }
        }

        if (!b64) throw new Error('Data gambar kosong — tidak dapat mengirim ke server lokal.');
        return realesrganUpscale(b64, opts.modelId, opts.onProgress);
    }

    if (!opts.apiKey?.trim()) throw new Error('API key kosong. Masukkan key terlebih dahulu.');

    if (provider === 'huggingface') {
        return hfUpscale(webglCanvas, targetW, targetH, opts);
    } else if (provider === 'replicate') {
        return replicateUpscale(webglCanvas, targetW, targetH, opts);
    } else if (provider === 'gemini') {
        return geminiImgUpscale(webglCanvas, targetW, targetH, opts);
    }
    throw new Error(`Provider tidak dikenal: ${provider}`);
}

/** Cek apakah server Real-ESRGAN lokal aktif */
async function realesrganCheck() {
    if (typeof realesrganPing !== 'function') return false;
    const r = await realesrganPing();
    return r.ok;
}

function aiSleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════
// UI Helper — Populate Model <select>
// ═══════════════════════════════════════════════════════
function aiPopulateModels(provider, selectEl) {
    selectEl.innerHTML = '';
    const models = AI_MODELS[provider] || [];
    models.forEach(m => {
        const opt   = document.createElement('option');
        opt.value   = m.id;
        opt.textContent = m.name;
        opt.title   = m.desc;
        selectEl.appendChild(opt);
    });
}

/** Call this from the UI on page init to restore saved keys */
function aiRestoreKeys() {
    // Auto-evict stale cached Gemini key when DEFAULT_GEMINI_KEY is updated
    try {
        const cachedVer = localStorage.getItem('scalerpro_gem_ver');
        if(cachedVer !== GEMINI_KEY_VERSION) {
            // Version mismatch — overwrite with new default key
            localStorage.setItem(LS_KEY_GEM, DEFAULT_GEMINI_KEY);
            localStorage.setItem('scalerpro_gem_ver', GEMINI_KEY_VERSION);
        } else if(!localStorage.getItem(LS_KEY_GEM)) {
            // No key saved yet — set default
            localStorage.setItem(LS_KEY_GEM, DEFAULT_GEMINI_KEY);
            localStorage.setItem('scalerpro_gem_ver', GEMINI_KEY_VERSION);
        }
    } catch(_) {}

    const hfEl  = document.getElementById('ai-hf-key');
    const repEl = document.getElementById('ai-rep-key');
    const gemEl = document.getElementById('ai-gem-key');
    const vidEl = document.getElementById('vid-gem-key');

    if(hfEl)  hfEl.value  = loadKey('huggingface');
    if(repEl) repEl.value = loadKey('replicate');
    if(gemEl) gemEl.value = loadKey('gemini');
    if(vidEl) vidEl.value = loadKey('gemini');
}

function aiPopulateModels(provider, selectEl) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    const models = AI_MODELS[provider] || [];
    models.forEach(m => {
        const opt   = document.createElement('option');
        opt.value   = m.id;
        opt.textContent = m.name;
        opt.title   = m.desc;
        selectEl.appendChild(opt);
    });
}
