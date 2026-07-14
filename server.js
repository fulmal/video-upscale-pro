/**
 * Video Upscale Pro — Node.js Express Server
 *
 * Responsibilities:
 *  1. Serve static frontend files
 *  2. Auto-spawn the Python AI server (Real-ESRGAN / Swin2SR)
 *  3. Proxy /api/* → Python server at localhost:5001
 *  4. Open browser automatically on startup
 */

'use strict';

const express      = require('express');
const path         = require('path');
const { spawn }    = require('child_process');
const http         = require('http');
const { createProxyMiddleware } = require('http-proxy-middleware');

// ─── Config ───────────────────────────────────────────────────────────────────
const PORT        = process.env.PORT || 3000;
const AI_PORT     = 5001;
const AI_BASE     = `http://127.0.0.1:${AI_PORT}`;
const ROOT        = __dirname;
const SERVER_DIR  = path.join(ROOT, 'realesrgan-server');
const PYTHON_BIN  = path.join(SERVER_DIR, '.venv', 'bin', 'python3');
const SERVER_PY   = path.join(SERVER_DIR, 'server.py');

// ─── App ──────────────────────────────────────────────────────────────────────
const app = express();

// Serve static files (HTML, CSS, JS, shaders)
app.use(express.static(ROOT, { index: 'index.html' }));

// ─── API Proxy → Python AI Server ─────────────────────────────────────────────
app.use('/api', createProxyMiddleware({
    target:      AI_BASE,
    changeOrigin: true,
    pathRewrite: { '^/api': '' },   // /api/ping → /ping, /api/upscale → /upscale
    on: {
        error: (err, req, res) => {
            console.error('[Proxy]', err.message);
            if (!res.headersSent) {
                res.status(502).json({ error: 'AI server tidak tersedia. Tunggu beberapa detik...' });
            }
        },
    },
}));

// ─── Status endpoint ──────────────────────────────────────────────────────────
app.get('/status', async (req, res) => {
    const aiOk = await pingAI();
    res.json({ node: 'ok', ai: aiOk ? 'ok' : 'starting', port: PORT, aiPort: AI_PORT });
});

// ─── Python AI Server Management ──────────────────────────────────────────────
let aiProcess = null;

function pingAI() {
    return new Promise(resolve => {
        const req = http.get(`${AI_BASE}/ping`, { timeout: 2000 }, res => {
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.on('timeout', () => { req.destroy(); resolve(false); });
    });
}

async function spawnAIServer() {
    // Don't spawn if already running
    if (await pingAI()) {
        console.log('✅ AI server sudah berjalan di port', AI_PORT);
        return;
    }

    console.log('🚀 Menjalankan AI server (Real-ESRGAN / Swin2SR)...');

    // Apply basicsr patch first (silently)
    const patchScript = path.join(SERVER_DIR, 'patch_basicsr.sh');
    try {
        const patch = spawn('/bin/bash', [patchScript], { cwd: SERVER_DIR });
        await new Promise(r => patch.on('close', r));
    } catch (_) { /* patch optional */ }

    // Spawn Python server
    aiProcess = spawn(PYTHON_BIN, [SERVER_PY], {
        cwd:   SERVER_DIR,
        stdio: 'pipe',
        env:   { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    // Pipe Python logs with prefix
    aiProcess.stdout.on('data', d => process.stdout.write(`[AI] ${d}`));
    aiProcess.stderr.on('data', d => process.stderr.write(`[AI] ${d}`));

    aiProcess.on('exit', (code, signal) => {
        if (code === 1) {
            // Might be "Address already in use" — another process already started server
            pingAI().then(ok => {
                if (ok) console.log('✅ AI server sudah tersedia (dari proses lain).');
            });
        }
        console.log(`[AI] Server berhenti (code=${code}, signal=${signal})`);
        aiProcess = null;
    });

    // Wait for AI server to be ready (max 35s, handles slow model load)
    const deadline = Date.now() + 35_000;
    while (Date.now() < deadline) {
        await sleep(1500);
        if (await pingAI()) {
            console.log('✅ AI server siap!');
            return;
        }
    }
    console.warn('⚠️  AI server belum siap setelah 35 detik. Real-ESRGAN mungkin tidak tersedia.');
}

// ─── Graceful Shutdown ────────────────────────────────────────────────────────
function shutdown(sig) {
    console.log(`\n[Node] Menerima ${sig}, menutup...`);
    if (aiProcess) {
        aiProcess.kill('SIGTERM');
    }
    process.exit(0);
}
process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

// ─── Helpers ──────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Start ────────────────────────────────────────────────────────────────────
async function main() {
    console.log('');
    console.log('🎬 Video Upscale Pro — Node.js Server');
    console.log('─────────────────────────────────────');

    // Start Express
    await new Promise(resolve => app.listen(PORT, '127.0.0.1', resolve));
    console.log(`🌐 Berjalan di http://localhost:${PORT}`);

    // Spawn AI server in background (non-blocking)
    spawnAIServer().catch(e => console.error('[AI spawn]', e));

    // Open browser
    try {
        const { default: open } = await import('open');
        await open(`http://localhost:${PORT}`);
    } catch (_) {
        console.log(`   Buka browser manual: http://localhost:${PORT}`);
    }

    console.log('   Tekan Ctrl+C untuk berhenti\n');
}

main().catch(err => { console.error(err); process.exit(1); });
