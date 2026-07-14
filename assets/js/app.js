/**
 * VideoScaler Pro — Main Application Logic
 * GPU-Accelerated Video Upscaler using WebGL
 */

// ══════════════════════════════════════════════
// Live Log Utility (shared by video + image)
// ══════════════════════════════════════════════
/**
 * Append a timestamped entry to a log panel.
 * @param {string} panelId  — 'vid-log' | 'img-log'
 * @param {string} msg      — text to show
 * @param {string} [type]   — 'info'(default)|'ok'|'err'|'warn'|'ai'|'gpu'
 */
function addLog(panelId, msg, type) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    const now = new Date();
    const ts  = now.toLocaleTimeString('id-ID', { hour12: false });
    const row = document.createElement('div');
    row.className = `log-row log-${type || 'info'}`;
    row.innerHTML = `<span class="log-ts">${ts}</span><span class="log-msg">${msg}</span>`;
    panel.appendChild(row);
    // Auto-scroll to bottom
    panel.scrollTop = panel.scrollHeight;
}

/** Clear a log panel and reset to initial state */
function clearLog(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.innerHTML = '<div class="log-row log-gpu"><span class="log-ts">--:--:--</span><span class="log-msg">Memulai proses...</span></div>';
}

// ══════════════════════════════════════════════
// WebGL Shader Sources
// ══════════════════════════════════════════════

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `
precision highp float;
uniform sampler2D u_img;
uniform vec2 u_ts;
uniform vec2 u_isz;
uniform float u_sharp, u_clar, u_vib, u_cmp, u_dens;
varying vec2 v_uv;

float cubic(float x){
    float ax=abs(x);
    if(ax<1.0) return 1.5*ax*ax*ax-2.5*ax*ax+1.0;
    if(ax<2.0) return -0.5*ax*ax*ax+2.5*ax*ax-4.0*ax+2.0;
    return 0.0;
}

vec4 bicubic(vec2 uv){
    vec2 p=uv*u_isz;
    vec2 f=fract(p);
    p=floor(p);
    vec4 col=vec4(0.0);
    float tw=0.0;
    for(int y=-1;y<=2;y++) for(int x=-1;x<=2;x++){
        float wx=cubic(float(x)-f.x);
        float wy=cubic(float(y)-f.y);
        float w=wx*wy;
        col+=texture2D(u_img,(p+vec2(float(x),float(y)))/u_isz)*w;
        tw+=w;
    }
    return col/max(tw,0.001);
}

vec4 enhance(vec2 uv, vec2 ts){
    vec4 cBi=texture2D(u_img,uv);
    vec4 cBc=bicubic(uv);
    float db=clamp(u_dens*2.0,0.0,1.0);
    vec4 c=mix(cBi,cBc,db);
    vec4 bl=vec4(0.0);
    float tw=0.0;
    for(int y=-2;y<=2;y++) for(int x=-2;x<=2;x++){
        float d=float(x*x+y*y);
        float w=exp(-d*0.35);
        bl+=texture2D(u_img,uv+vec2(float(x),float(y))*ts)*w;
        tw+=w;
    }
    bl/=tw;
    vec3 d2=c.rgb-bl.rgb;
    vec3 r=c.rgb+d2*u_sharp;
    r+=d2*u_clar*0.55;
    vec3 e=texture2D(u_img,uv+vec2(0,-ts.y)).rgb
           +texture2D(u_img,uv+vec2(0,ts.y)).rgb
           +texture2D(u_img,uv+vec2(-ts.x,0)).rgb
           +texture2D(u_img,uv+vec2(ts.x,0)).rgb
           -4.0*c.rgb;
    r+=e*u_clar*0.28;
    if(u_dens>0.5){
        vec3 ne=texture2D(u_img,uv+vec2( ts.x,-ts.y)*0.5).rgb;
        vec3 nw=texture2D(u_img,uv+vec2(-ts.x,-ts.y)*0.5).rgb;
        vec3 se=texture2D(u_img,uv+vec2( ts.x, ts.y)*0.5).rgb;
        vec3 sw=texture2D(u_img,uv+vec2(-ts.x, ts.y)*0.5).rgb;
        vec3 hf=c.rgb-(ne+nw+se+sw)*0.25;
        float xd=clamp((u_dens-0.5)*2.0,0.0,1.0);
        r+=hf*xd*0.65;
    }
    r=(r-0.5)*1.06+0.5;
    float lm=dot(r,vec3(.2126,.7152,.0722));
    vec3 gr=vec3(lm);
    float sat=length(r-gr);
    r=mix(gr,r,1.0+u_vib*(1.0-min(sat*4.0,1.0)));
    return vec4(clamp(r,0.0,1.0),c.a);
}

void main(){
    vec2 uv=v_uv;
    if(u_cmp>0.5){
        if(uv.x<0.497){
            gl_FragColor=texture2D(u_img,vec2(uv.x/0.497,uv.y));
            return;
        } else if(uv.x>0.503){
            uv=vec2((uv.x-0.503)/0.497,uv.y);
        } else { gl_FragColor=vec4(1.0); return; }
    }
    gl_FragColor=enhance(uv,u_ts);
}`;

// ══════════════════════════════════════════════
// App State
// ══════════════════════════════════════════════

let step = 1, vidFile = null, vidUrl = null, vW = 0, vH = 0, vDur = 0;
let selRes = null, selFmt = 'webm', isProc = false;
let resBlob = null, resUrl = null, resExt = 'webm';

// WebGL references
let gl = null;
let cmpRef = null;
let cmpAnimId = null;

const hV   = document.getElementById('hV');
const pC   = document.getElementById('pC');
const cmpC = document.getElementById('cmpC');

const RES = [
    { n: '720p',  w: 1280,  h: 720  },
    { n: '1080p', w: 1920,  h: 1080 },
    { n: '2K',    w: 2560,  h: 1440 },
    { n: '4K',    w: 3840,  h: 2160 },
];

// ══════════════════════════════════════════════
// Step Navigation
// ══════════════════════════════════════════════

function goS(s) {
    if ((s < step && !isProc) || s === 1) setS(s);
}

function setS(s) {
    step = s;
    document.querySelectorAll('.sp').forEach(p => p.classList.remove('active'));
    document.getElementById('step' + s).classList.add('active');
    document.querySelectorAll('.si').forEach(i => {
        const n = +i.dataset.step;
        i.classList.remove('active', 'done');
        if (n === s)    i.classList.add('active');
        else if (n < s) i.classList.add('done');
    });
    document.querySelectorAll('.sc').forEach((c, i) => c.classList.toggle('done', i < s - 1));
}

// ══════════════════════════════════════════════
// File Upload
// ══════════════════════════════════════════════

const uz = document.getElementById('uz');
const fi = document.getElementById('fi');

uz.addEventListener('click',    e => { if (e.target.tagName !== 'BUTTON') fi.click(); });
uz.addEventListener('dragover', e => { e.preventDefault(); uz.classList.add('drag'); });
uz.addEventListener('dragleave',() => uz.classList.remove('drag'));
uz.addEventListener('drop', e => {
    e.preventDefault();
    uz.classList.remove('drag');
    const f = e.dataTransfer.files;
    if (f.length && f[0].type.startsWith('video/')) handleFile(f[0]);
    else toast('File harus video!', 'err');
});
fi.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(f) {
    vidFile = f;
    if (vidUrl) URL.revokeObjectURL(vidUrl);
    vidUrl = URL.createObjectURL(f);

    const pv = document.getElementById('pv');
    pv.src = vidUrl;
    pv.onloadedmetadata = () => {
        vW = pv.videoWidth; vH = pv.videoHeight; vDur = pv.duration;
        document.getElementById('fn2').textContent  = f.name;
        document.getElementById('sRes').textContent = `${vW} × ${vH}`;
        document.getElementById('sDur').textContent = fDur(vDur);
        document.getElementById('sSiz').textContent = fSiz(f.size);
        buildRes();
        checkCodec();
        setS(2);
        setTimeout(initCompare, 300);
    };
    pv.onerror = () => toast('Gagal memuat video!', 'err');
}

function rmVid() {
    if (vidUrl) URL.revokeObjectURL(vidUrl);
    vidUrl = null; vidFile = null; vW = 0; vH = 0; vDur = 0; fi.value = '';
    document.getElementById('pv').src = '';
    stopCmp();
    setS(1);
}

// ══════════════════════════════════════════════
// Resolution Options
// ══════════════════════════════════════════════

function buildRes() {
    const c = document.getElementById('resOpts');
    c.innerHTML = '';
    selRes = null;
    const srcPx = vW * vH;

    RES.forEach(r => {
        const d   = document.createElement('div');
        d.className = 'rp';
        const tP   = r.w * r.h;
        const dis  = tP < srcPx * 0.95;
        const same = Math.abs(tP - srcPx) / srcPx < 0.05;
        if (dis || same) d.classList.add('dis');

        let b = '';
        if (!dis && !same)  b = `<div class="ru">↑ ${(tP / srcPx).toFixed(1)}x</div>`;
        else if (same)      b = `<div class="ru" style="background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.2);color:var(--orange)">Sama</div>`;
        else                b = `<div class="ru" style="background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.2);color:var(--red)">↓ Down</div>`;

        d.innerHTML = `<div class="rn">${r.n}</div><div class="rd">${r.w}×${r.h}</div>${b}`;
        d.onclick = () => {
            if (dis) return;
            c.querySelectorAll('.rp').forEach(o => o.classList.remove('sel'));
            d.classList.add('sel');
            selRes = r;
            upC();
        };
        c.appendChild(d);
    });

    const auto = RES.find(r => r.w * r.h > srcPx);
    if (auto) {
        c.children[RES.indexOf(auto)].click();
    } else {
        const fb = RES.find(r => r.h >= 1080) || RES[0];
        const i  = RES.indexOf(fb);
        if (!c.children[i].classList.contains('dis')) c.children[i].click();
    }
}

// ══════════════════════════════════════════════
// Codec Check
// ══════════════════════════════════════════════

function checkCodec() {
    const n  = document.getElementById('cn');
    const wt = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mt = ['video/mp4;codecs=h264,aac',  'video/mp4;codecs=h264',  'video/mp4'];
    const ws = wt.find(t => MediaRecorder.isTypeSupported(t));
    const ms = mt.find(t => MediaRecorder.isTypeSupported(t));

    n.style.display = 'block';
    n.innerHTML = `<b>WebM:</b> ${ws || '✕'} &nbsp;|&nbsp; <b>MP4:</b> ${ms || '✕'}`;
    if (!ws && ms) selF(document.querySelector('.fo2[data-fmt="mp4"]'));
    if (!ws && !ms) {
        toast('Browser tidak support video recording!', 'err');
        document.getElementById('startBtn').disabled = true;
    }
}

// ══════════════════════════════════════════════
// Format & Slider UI
// ══════════════════════════════════════════════

function selF(el) {
    document.querySelectorAll('.fo2').forEach(f => f.classList.remove('sel'));
    el.classList.add('sel');
    selFmt = el.dataset.fmt;
}

function upC() {
    document.getElementById('shV').textContent = document.getElementById('shS').value + '%';
    document.getElementById('clV').textContent = document.getElementById('clS').value + '%';
    document.getElementById('viV').textContent = document.getElementById('viS').value + '%';
    document.getElementById('dnV').textContent = document.getElementById('dnS').value + '%';
    renderCmp();
}

// ══════════════════════════════════════════════
// WebGL Helpers
// ══════════════════════════════════════════════

function mkShader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error('Shader error:', gl.getShaderInfoLog(s));
        return null;
    }
    return s;
}

function initGL(canvas) {
    const c = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: false });
    if (!c) return null;

    const vs = mkShader(c, c.VERTEX_SHADER,   VERT);
    const fs = mkShader(c, c.FRAGMENT_SHADER, FRAG);
    const p  = c.createProgram();
    c.attachShader(p, vs); c.attachShader(p, fs); c.linkProgram(p);
    if (!c.getProgramParameter(p, c.LINK_STATUS)) {
        console.error(c.getProgramInfoLog(p));
        return null;
    }
    c.useProgram(p);

    // Full-screen quad
    const buf = c.createBuffer();
    c.bindBuffer(c.ARRAY_BUFFER, buf);
    c.bufferData(c.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), c.STATIC_DRAW);
    const ap = c.getAttribLocation(p, 'a_pos');
    c.enableVertexAttribArray(ap);
    c.vertexAttribPointer(ap, 2, c.FLOAT, false, 0, 0);

    // Texture
    const t = c.createTexture();
    c.bindTexture(c.TEXTURE_2D, t);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_S,     c.CLAMP_TO_EDGE);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_T,     c.CLAMP_TO_EDGE);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MIN_FILTER, c.LINEAR);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MAG_FILTER, c.LINEAR);
    c.pixelStorei(c.UNPACK_FLIP_Y_WEBGL, true);

    return {
        gl: c, prog: p, tex: t,
        u: {
            img:   c.getUniformLocation(p, 'u_img'),
            ts:    c.getUniformLocation(p, 'u_ts'),
            isz:   c.getUniformLocation(p, 'u_isz'),
            sharp: c.getUniformLocation(p, 'u_sharp'),
            clar:  c.getUniformLocation(p, 'u_clar'),
            vib:   c.getUniformLocation(p, 'u_vib'),
            cmp:   c.getUniformLocation(p, 'u_cmp'),
            dens:  c.getUniformLocation(p, 'u_dens'),
        }
    };
}

function renderGL(ref, video, w, h, opts) {
    const g = ref.gl;
    g.viewport(0, 0, w, h);
    g.bindTexture(g.TEXTURE_2D, ref.tex);
    g.texImage2D(g.TEXTURE_2D, 0, g.RGBA, g.RGBA, g.UNSIGNED_BYTE, video);
    g.uniform1i(ref.u.img,   0);
    g.uniform2f(ref.u.ts,    1 / w,        1 / h);
    g.uniform2f(ref.u.isz,   opts.iw || w, opts.ih || h);
    g.uniform1f(ref.u.sharp, opts.sharp || 0);
    g.uniform1f(ref.u.clar,  opts.clar  || 0);
    g.uniform1f(ref.u.vib,   opts.vib   || 0);
    g.uniform1f(ref.u.cmp,   opts.cmp ? 1 : 0);
    g.uniform1f(ref.u.dens,  opts.dens  || 0);
    g.drawArrays(g.TRIANGLE_STRIP, 0, 4);
}

// ══════════════════════════════════════════════
// Comparison Preview
// ══════════════════════════════════════════════

function initCompare() {
    stopCmp();
    cmpC.width  = 800;
    cmpC.height = Math.round(800 * (vH / vW));
    cmpRef = initGL(cmpC);
    if (!cmpRef) return;
    renderCmp();
}

function renderCmp() {
    if (!cmpRef) return;
    const pv = document.getElementById('pv');
    if (!pv.videoWidth) return;
    renderGL(cmpRef, pv, cmpC.width, cmpC.height, {
        sharp: +document.getElementById('shS').value / 100,
        clar:  +document.getElementById('clS').value / 100,
        vib:   +document.getElementById('viS').value / 100,
        dens:  +document.getElementById('dnS').value / 100,
        iw:    pv.videoWidth,
        ih:    pv.videoHeight,
        cmp:   true,
    });
}

function startCmpLoop() {
    const pv = document.getElementById('pv');
    function loop() {
        if (pv.paused || pv.ended || step !== 2) { cmpAnimId = null; return; }
        renderCmp();
        cmpAnimId = requestAnimationFrame(loop);
    }
    loop();
}

function stopCmp() {
    if (cmpAnimId) { cancelAnimationFrame(cmpAnimId); cmpAnimId = null; }
}

document.getElementById('pv').addEventListener('play',   () => { if (step === 2) startCmpLoop(); });
document.getElementById('pv').addEventListener('pause',  () => { stopCmp(); renderCmp(); });
document.getElementById('pv').addEventListener('seeked', () => { if (step === 2) renderCmp(); });

// ══════════════════════════════════════════════
// Processing (Upscale)
// ══════════════════════════════════════════════

async function startProc() {
    if (!selRes) { toast('Pilih target resolusi!', 'err'); return; }
    if (selRes.w <= vW && selRes.h <= vH) { toast('Target harus lebih besar dari source!', 'err'); return; }

    setS(3); isProc = true; stopCmp();

    const video = hV;
    video.src = vidUrl; video.muted = true; video.playsInline = true; video.currentTime = 0;

    try {
        await new Promise((ok, no) => {
            video.onloadeddata = ok;
            video.onerror      = no;
            setTimeout(no, 15000);
        });
    } catch (e) { toast('Gagal memuat video!', 'err'); setS(2); isProc = false; return; }

    // Output dimensions (maintain aspect ratio, even numbers)
    const srcR = vW / vH;
    let fW = selRes.w, fH = selRes.h;
    if (fW / fH > srcR) fW = Math.round(fH * srcR);
    else                 fH = Math.round(fW / srcR);
    fW = fW % 2 ? fW + 1 : fW;
    fH = fH % 2 ? fH + 1 : fH;

    // Init WebGL on process canvas
    pC.style.display = 'block';
    pC.width  = fW; pC.height = fH;
    gl = initGL(pC);
    if (!gl) { toast('WebGL tidak tersedia!', 'err'); setS(2); isProc = false; pC.style.display = 'none'; return; }

    document.getElementById('pS').innerHTML = '<strong>Memproses...</strong> GPU shader sedang bekerja';
    clearLog('vid-log');
    addLog('vid-log', `🎬 Video: ${vW}×${vH} → ${fW}×${fH}`, 'gpu');
    addLog('vid-log', `⚙️ Sharp: ${Math.round(shV*100)}% | Clarity: ${Math.round(clV*100)}% | Vibrance: ${Math.round(viV*100)}%`, 'info');

    // Codec selection
    const prefMp4 = selFmt === 'mp4';
    const types   = prefMp4
        ? ['video/mp4;codecs=h264,aac', 'video/mp4;codecs=h264', 'video/mp4',
           'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp9', 'video/webm']
        : ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8',
           'video/webm', 'video/mp4;codecs=h264,aac', 'video/mp4'];
    const mime = types.find(t => MediaRecorder.isTypeSupported(t));
    if (!mime) { toast('Codec tidak didukung!', 'err'); setS(2); isProc = false; pC.style.display = 'none'; return; }
    resExt = mime.includes('mp4') ? 'mp4' : 'webm';
    addLog('vid-log', `🎞️ Codec: ${mime.split(';')[0]} (.${resExt})`, 'info');

    // Bitrate scaling by resolution
    const qPct = +document.getElementById('qS').value;
    let bps = fW >= 3840 ? qPct * 4e5
            : fW >= 2560 ? qPct * 2e5
            : fW >= 1920 ? qPct * 1.2e5
            :              qPct * 8e4;
    bps = Math.max(5e5, Math.round(bps));

    // MediaRecorder setup
    const chunks = [];
    let rec;
    try {
        rec = new MediaRecorder(pC.captureStream(30), { mimeType: mime, videoBitsPerSecond: bps });
    } catch (e) {
        try { rec = new MediaRecorder(pC.captureStream(30), { mimeType: mime }); }
        catch (e2) { toast('Recorder gagal!', 'err'); setS(2); isProc = false; pC.style.display = 'none'; return; }
    }

    const recDone = new Promise(res => {
        rec.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
        rec.onstop          = () => { try { res(new Blob(chunks, { type: mime })); } catch (e) { res(null); } };
        rec.onerror         = () => res(null);
    });
    rec.start(1000);

    // Read enhancement values once
    const shV = +document.getElementById('shS').value / 100;
    const clV = +document.getElementById('clS').value / 100;
    const viV = +document.getElementById('viS').value / 100;
    const dnV = +document.getElementById('dnS').value / 100;

    const totalDur  = video.duration;
    const startTime = performance.now();
    let   done      = false;

    await video.play();

    function loop() {
        if (!isProc || done) return;
        if (video.paused || video.ended || video.currentTime >= totalDur - 0.05) { finish(); return; }

        renderGL(gl, video, fW, fH, { sharp: shV, clar: clV, vib: viV, dens: dnV, iw: vW, ih: vH, cmp: false });

        const prog    = Math.min(video.currentTime / totalDur * 100, 99.9);
        const elapsed = (performance.now() - startTime) / 1000;
        const eta     = prog > 1 ? (elapsed / prog) * (100 - prog) : 0;

        document.getElementById('pP').textContent      = Math.round(prog) + '%';
        document.getElementById('pF').style.width      = prog + '%';
        document.getElementById('fI').textContent      = Math.round(video.currentTime * 30);
        document.getElementById('tI').textContent      = fDur(video.currentTime);
        document.getElementById('eI').textContent      = eta > 0 ? fDur(eta) : '—';

        // Log setiap ~5% progress
        const progInt = Math.floor(prog / 5) * 5;
        if (progInt > 0 && progInt !== (window._vidLastLogPct || 0)) {
            window._vidLastLogPct = progInt;
            addLog('vid-log',
                `Frame ${Math.round(video.currentTime * 30)} | ${fDur(video.currentTime)} | ${Math.round(prog)}% | ETA: ${eta > 0 ? fDur(eta) : '—'}`,
                'gpu');
        }

        requestAnimationFrame(loop);
    }

    async function finish() {
        if (done) return;
        done = true;
        document.getElementById('pP').textContent = '99%';
        document.getElementById('pF').style.width = '99%';
        document.getElementById('pS').innerHTML   = '<strong>Finalisasi...</strong> Menunggu encoder';
        addLog('vid-log', '⏳ Finalisasi — menunggu encoder...', 'warn');

        video.pause();
        await sleep(1000);
        if (rec.state === 'recording') rec.stop();

        const blob = await recDone;
        pC.style.display = 'none';

        if (!blob || blob.size < 100) { toast('Recording gagal!', 'err'); setS(2); isProc = false; return; }
        if (resUrl) URL.revokeObjectURL(resUrl);

        resBlob = blob;
        resUrl  = URL.createObjectURL(blob);
        const total = (performance.now() - startTime) / 1000;

        document.getElementById('pP').textContent = '100%';
        document.getElementById('pF').style.width = '100%';
        document.getElementById('sF').textContent = gRL(vW, vH);
        document.getElementById('sT').textContent = gRL(fW, fH);
        document.getElementById('sSz').textContent = fSiz(blob.size);
        document.getElementById('sTi').textContent = fDur(total);
        document.getElementById('upT').textContent = `${gRL(vW,vH)} → ${gRL(fW,fH)} | Unsharp + Clarity + Vibrance`;
        addLog('vid-log', `✅ Selesai! ${gRL(vW,vH)} → ${gRL(fW,fH)} | ${fSiz(blob.size)} | ${fDur(total)}`, 'ok');

        window._vidLastLogPct = 0;
        isProc = false;
        setTimeout(() => setS(4), 500);
    }

    loop();
}

function cancelProc() {
    isProc = false;
    hV.pause(); hV.src = '';
    pC.style.display = 'none';
    addLog('vid-log', '🛑 Proses dibatalkan oleh pengguna', 'err');
    toast('Dibatalkan', 'err');
    setS(2);
}

// ══════════════════════════════════════════════
// Download
// ══════════════════════════════════════════════

function dlResult() {
    if (!resUrl) return;
    const a  = document.createElement('a');
    a.href   = resUrl;
    const bn = vidFile ? vidFile.name.replace(/\.[^.]+$/, '') : 'video';
    a.download = `${bn}_${selRes ? selRes.n : 'hd'}_enhanced.${resExt}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    toast('Download dimulai!');
}

// ══════════════════════════════════════════════
// Reset
// ══════════════════════════════════════════════

function resetAll() {
    if (vidUrl) URL.revokeObjectURL(vidUrl);
    if (resUrl) URL.revokeObjectURL(resUrl);
    vidUrl = null; resUrl = null; resBlob = null;
    vidFile = null; vW = 0; vH = 0; vDur = 0;
    selRes = null; isProc = false;
    fi.value = '';
    document.getElementById('pv').src = '';
    hV.src = '';
    pC.style.display = 'none';
    document.getElementById('pP').textContent = '0%';
    document.getElementById('pF').style.width = '0%';
    document.getElementById('fI').textContent = '0';
    document.getElementById('tI').textContent = '0:00';
    document.getElementById('eI').textContent = '—';
    setS(1);
}

// ══════════════════════════════════════════════
// Utility Helpers
// ══════════════════════════════════════════════

function fDur(s) {
    if (!s || isNaN(s)) return '0:00';
    const m  = Math.floor(s / 60);
    const sc = Math.floor(s % 60);
    return m + ':' + String(sc).padStart(2, '0');
}

function fSiz(b) {
    if (!b) return '—';
    if (b < 1024)       return b + ' B';
    if (b < 1048576)    return (b / 1024).toFixed(1) + ' KB';
    if (b < 1073741824) return (b / 1048576).toFixed(1) + ' MB';
    return (b / 1073741824).toFixed(2) + ' GB';
}

function gRL(w, h) {
    if (h >= 2160) return '4K';
    if (h >= 1440) return '2K';
    if (h >= 1080) return '1080p';
    if (h >= 720)  return '720p';
    return w + '×' + h;
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function toast(m, t = 'ok') {
    const e  = document.getElementById('toast');
    const ic = t === 'err'
        ? '<svg viewBox="0 0 24 24" fill="none" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    e.className = 'toast ' + (t === 'err' ? 'err' : '');
    e.innerHTML = ic + ' ' + m;
    e.classList.add('show');
    clearTimeout(e._t);
    e._t = setTimeout(() => e.classList.remove('show'), 4000);
}
