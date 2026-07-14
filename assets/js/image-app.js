/**
 * ScalerPro — Image Upscale Extreme
 * Multi-pass Bicubic + Bilateral Denoise + HF Synthesis GPU upscaler
 */

// ═══════════════════════════════════════════════════════
// Shader Sources
// ═══════════════════════════════════════════════════════

const IMG_VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const IMG_FRAG = `
precision highp float;
uniform sampler2D u_img;
uniform vec2 u_ts;    /* source texel size: 1/srcW, 1/srcH */
uniform vec2 u_isz;   /* source image dimensions */
uniform float u_sharp, u_clar, u_vib, u_dens, u_noise;
varying vec2 v_uv;

/* ── Bilateral Denoise (edge-preserving smoothing) ── */
vec4 bilateralDenoise(vec2 uv){
    if(u_noise < 0.01) return texture2D(u_img, uv);
    vec4 center = texture2D(u_img, uv);
    vec4 sum = vec4(0.0); float tot = 0.0;
    float sigS = u_noise * 5.0 + 0.5;
    float sigR = 0.10 + u_noise * 0.12;
    for(int y = -2; y <= 2; y++) for(int x = -2; x <= 2; x++){
        vec4 s = texture2D(u_img, uv + vec2(float(x), float(y)) * u_ts);
        float wS = exp(-float(x*x + y*y) / (2.0 * sigS * sigS));
        float diff = length(s.rgb - center.rgb);
        float wR = exp(-(diff * diff) / (2.0 * sigR * sigR));
        float w = wS * wR;
        sum += s * w; tot += w;
    }
    return sum / max(tot, 0.001);
}

/* ── Catmull-Rom Cubic Kernel ── */
float cb(float x){
    float ax = abs(x);
    if(ax < 1.0) return  1.5*ax*ax*ax - 2.5*ax*ax + 1.0;
    if(ax < 2.0) return -0.5*ax*ax*ax + 2.5*ax*ax - 4.0*ax + 2.0;
    return 0.0;
}

/* ── Bicubic 4×4 Catmull-Rom Sampler ── */
vec4 bicubic(vec2 uv){
    vec2 p = uv * u_isz, f = fract(p); p = floor(p);
    vec4 col = vec4(0.0); float tw = 0.0;
    for(int y = -1; y <= 2; y++) for(int x = -1; x <= 2; x++){
        float w = cb(float(x) - f.x) * cb(float(y) - f.y);
        col += texture2D(u_img, (p + vec2(float(x), float(y))) / u_isz) * w;
        tw  += w;
    }
    return col / max(tw, 0.001);
}

void main(){
    vec2 uv = v_uv;

    /* 1. Bilateral denoise source */
    vec4 dn = bilateralDenoise(uv);
    vec4 bi = texture2D(u_img, uv);
    vec4 bc = bicubic(uv);

    /* 2. Blend: bilinear → denoise → bicubic based on params */
    float db = clamp(u_dens  * 2.0, 0.0, 1.0);
    float nb = clamp(u_noise * 5.0, 0.0, 1.0);
    vec4 c = mix(mix(bi, dn, nb), bc, db);

    /* 3. Gaussian blur for Unsharp Mask (in source pixel space) */
    vec4 bl = vec4(0.0); float tw = 0.0;
    for(int y = -2; y <= 2; y++) for(int x = -2; x <= 2; x++){
        float d = float(x*x + y*y);
        float w = exp(-d * 0.35);
        bl += texture2D(u_img, uv + vec2(float(x), float(y)) * u_ts) * w;
        tw += w;
    }
    bl /= tw;

    /* 4. Unsharp Mask + Clarity */
    vec3 d2 = c.rgb - bl.rgb;
    vec3 r   = c.rgb + d2 * u_sharp;
    r += d2 * u_clar * 0.55;

    /* 5. Laplacian edge clarity */
    vec3 e = texture2D(u_img, uv + vec2(0.0,  -u_ts.y)).rgb
           + texture2D(u_img, uv + vec2(0.0,   u_ts.y)).rgb
           + texture2D(u_img, uv + vec2(-u_ts.x, 0.0)).rgb
           + texture2D(u_img, uv + vec2( u_ts.x, 0.0)).rgb
           - 4.0 * c.rgb;
    r += e * u_clar * 0.28;

    /* 6. High-frequency sub-pixel detail synthesis (Density > 50%) */
    if(u_dens > 0.5){
        vec3 ne = texture2D(u_img, uv + vec2( u_ts.x, -u_ts.y) * 0.5).rgb;
        vec3 nw = texture2D(u_img, uv + vec2(-u_ts.x, -u_ts.y) * 0.5).rgb;
        vec3 se = texture2D(u_img, uv + vec2( u_ts.x,  u_ts.y) * 0.5).rgb;
        vec3 sw = texture2D(u_img, uv + vec2(-u_ts.x,  u_ts.y) * 0.5).rgb;
        vec3 hf = c.rgb - (ne + nw + se + sw) * 0.25;
        float xd = clamp((u_dens - 0.5) * 2.0, 0.0, 1.0);
        r += hf * xd * 0.85;
    }

    /* 7. Contrast micro-boost */
    r = (r - 0.5) * 1.04 + 0.5;

    /* 8. Vibrance (selective saturation) */
    float lm  = dot(r, vec3(.2126, .7152, .0722));
    vec3  gr  = vec3(lm);
    float sat = length(r - gr);
    r = mix(gr, r, 1.0 + u_vib * (1.0 - min(sat * 4.0, 1.0)));

    gl_FragColor = vec4(clamp(r, 0.0, 1.0), c.a);
}`;

// ═══════════════════════════════════════════════════════
// Fractal IFS Shader (Genuine Fractals-inspired)
// ═══════════════════════════════════════════════════════
/**
 * Fractal IFS Upscaling Engine
 * Inspired by: Partitioned Iterated Function Systems (PIFS)
 * as used in Genuine Fractals / ON1 Resize.
 */
const IMG_FRAG_FRACTAL = `
precision highp float;
uniform sampler2D u_img;
uniform vec2 u_ts;      /* source texel size */
uniform vec2 u_isz;     /* source image size */
uniform float u_sharp, u_clar, u_vib, u_dens, u_noise, u_frac_detail;
varying vec2 v_uv;

float cr(float x){
    float a=abs(x);
    if(a<1.0) return 1.5*a*a*a - 2.5*a*a + 1.0;
    if(a<2.0) return -0.5*a*a*a + 2.5*a*a - 4.0*a + 2.0;
    return 0.0;
}
vec4 bicubicSample(vec2 uv){
    vec2 p=uv*u_isz; vec2 f=fract(p); p=floor(p);
    vec4 c=vec4(0.0); float tw=0.0;
    for(int y=-1;y<=2;y++) for(int x=-1;x<=2;x++){
        float w=cr(float(x)-f.x)*cr(float(y)-f.y);
        c+=texture2D(u_img,(p+vec2(float(x),float(y)))/u_isz)*w;
        tw+=w;
    }
    return c/max(tw,0.001);
}
vec2 ifsTransform(vec2 d, int t){
    if(t==1) return vec2(-d.y, d.x); if(t==2) return vec2(-d.x,-d.y);
    if(t==3) return vec2( d.y,-d.x); if(t==4) return vec2(-d.x, d.y);
    if(t==5) return vec2( d.x,-d.y); if(t==6) return vec2( d.y, d.x);
    if(t==7) return vec2(-d.y,-d.x); return d;
}
float luma(vec3 c){ return dot(c,vec3(0.2126,0.7152,0.0722)); }
vec4 fractalIFS(vec2 uv){
    vec4 center = bicubicSample(uv);
    float cl = luma(center.rgb);
    vec4 fracSum = vec4(0.0); float fracTot = 0.0;
    for(int dy=-2; dy<=2; dy++) for(int dx=-2; dx<=2; dx++){
        if(dx==0 && dy==0) continue;
        vec2 domainUV = uv + vec2(float(dx),float(dy)) * u_ts * 2.0;
        if(domainUV.x<0.0||domainUV.x>1.0||domainUV.y<0.0||domainUV.y>1.0) continue;
        for(int t=0; t<8; t++){
            vec4 dom = (bicubicSample(domainUV + ifsTransform(vec2(-0.5,-0.5)*u_ts, t)) +
                        bicubicSample(domainUV + ifsTransform(vec2( 0.5,-0.5)*u_ts, t)) +
                        bicubicSample(domainUV + ifsTransform(vec2(-0.5, 0.5)*u_ts, t)) +
                        bicubicSample(domainUV + ifsTransform(vec2( 0.5, 0.5)*u_ts, t)))*0.25;
            float dl = luma(dom.rgb);
            float scale = clamp((dl > 0.001) ? (cl / dl) : 1.0, 0.5, 2.0);
            vec4 candidate = dom * scale;
            float w = exp(-abs(luma(candidate.rgb) - cl) * 12.0) * exp(-float(dx*dx+dy*dy)*0.18);
            fracSum += candidate * w; fracTot += w;
        }
    }
    vec4 fracResult = fracTot > 0.001 ? fracSum/fracTot : center;
    vec3 gx = texture2D(u_img,uv+vec2( u_ts.x,0.0)).rgb - texture2D(u_img,uv+vec2(-u_ts.x,0.0)).rgb;
    vec3 gy = texture2D(u_img,uv+vec2(0.0, u_ts.y)).rgb - texture2D(u_img,uv+vec2(0.0,-u_ts.y)).rgb;
    return mix(bicubicSample(uv), fracResult, (1.0 - clamp(length(gx)+length(gy), 0.0, 1.0) * 0.75) * u_frac_detail);
}
vec4 bilateralDenoise(vec2 uv){
    if(u_noise < 0.01) return texture2D(u_img, uv);
    vec4 center=texture2D(u_img,uv);
    vec4 sum=vec4(0.0); float tot=0.0;
    float sigS=u_noise*5.0+0.5, sigR=0.10+u_noise*0.12;
    for(int y=-2;y<=2;y++) for(int x=-2;x<=2;x++){
        vec4 s=texture2D(u_img,uv+vec2(float(x),float(y))*u_ts);
        float wS=exp(-float(x*x+y*y)/(2.0*sigS*sigS));
        float diff=length(s.rgb-center.rgb);
        float w=wS*exp(-(diff*diff)/(2.0*sigR*sigR));
        sum+=s*w; tot+=w;
    }
    return sum/max(tot,0.001);
}
void main(){
    vec2 uv = v_uv;

    /* 1. Bilateral denoise */
    vec4 dn = bilateralDenoise(uv);
    float nb = clamp(u_noise*5.0,0.0,1.0);
    vec4 base = mix(texture2D(u_img,uv), dn, nb);

    /* 2. Fractal IFS upscale (replaces bicubic for smooth regions) */
    vec4 c = mix(base, fractalIFS(uv), clamp(u_dens*2.0,0.0,1.0));

    /* 3. Gaussian blur for Unsharp Mask */
    vec4 bl=vec4(0.0); float tw=0.0;
    for(int y=-2;y<=2;y++) for(int x=-2;x<=2;x++){
        float w=exp(-float(x*x+y*y)*0.35);
        bl+=texture2D(u_img,uv+vec2(float(x),float(y))*u_ts)*w;
        tw+=w;
    }
    bl/=tw;

    /* 4. Unsharp mask + Clarity */
    vec3 diff=c.rgb-bl.rgb;
    vec3 r=c.rgb+diff*u_sharp;
    r+=diff*u_clar*0.55;

    /* 5. Laplacian edge clarity */
    vec3 e=texture2D(u_img,uv+vec2(0.0,-u_ts.y)).rgb
          +texture2D(u_img,uv+vec2(0.0, u_ts.y)).rgb
          +texture2D(u_img,uv+vec2(-u_ts.x,0.0)).rgb
          +texture2D(u_img,uv+vec2( u_ts.x,0.0)).rgb
          -4.0*c.rgb;
    r+=e*u_clar*0.28;

    /* 6. High-frequency fractal detail synthesis */
    if(u_frac_detail > 0.05 && u_dens > 0.5){
        vec2 fp = fract(uv*u_isz);
        float h =fract(sin(dot(fp,vec2(127.1,311.7)))*43758.5453);
        float h2=fract(sin(dot(fp,vec2(269.5,183.3)))*12345.6789);
        vec3 det=(texture2D(u_img,uv+(vec2(h,h2)-0.5)*u_ts*0.5).rgb
                 +texture2D(u_img,uv+(vec2(h2,h)-0.5)*u_ts*0.25).rgb)*0.5
                 -texture2D(u_img,uv).rgb;
        r+=det*u_frac_detail*clamp((u_dens-0.5)*2.0,0.0,1.0)*0.3;
    }

    /* 7. Contrast micro-boost */
    r=(r-0.5)*1.04+0.5;

    /* 8. Vibrance */
    float lm=dot(r,vec3(.2126,.7152,.0722));
    vec3 gr=vec3(lm);
    float sat=length(r-gr);
    r=mix(gr,r,1.0+u_vib*(1.0-min(sat*4.0,1.0)));

    gl_FragColor=vec4(clamp(r,0.0,1.0),c.a);
}`;

// ═══════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════
let imgFile = null, imgEl = null, imgSrcW = 0, imgSrcH = 0;
let imgCmpRef = null, imgDivPct = 50, imgIsDrag = false;
let imgSelTargetW = 0, imgSelTargetH = 0, imgSelLabel = '4K';
let imgSelFmt = 'png', imgResultUrl = null, imgIsProc = false;

// Engine state: 'bicubic' | 'fractal'
let imgEngine = 'bicubic';

// AI state
let aiEnabled   = false;
let aiProvider  = 'huggingface'; // 'huggingface' | 'replicate'
let aiResultWasUsed = false;

// ═══════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════
const imgBeforeC = document.getElementById('img-before-c');
const imgAfterC  = document.getElementById('img-after-c');
const imgCmpWrap = document.getElementById('img-cmp-wrap');
const imgDivider = document.getElementById('img-divider');
const imgUz      = document.getElementById('img-uz');
const imgFi      = document.getElementById('img-fi');

// ═══════════════════════════════════════════════════════
// File Upload
// ═══════════════════════════════════════════════════════
imgUz.addEventListener('click',    e => { if(e.target.tagName !== 'BUTTON') imgFi.click(); });
imgUz.addEventListener('dragover', e => { e.preventDefault(); imgUz.classList.add('drag'); });
imgUz.addEventListener('dragleave',() => imgUz.classList.remove('drag'));
imgUz.addEventListener('drop', e => {
    e.preventDefault(); imgUz.classList.remove('drag');
    const f = e.dataTransfer.files;
    if(f.length && f[0].type.startsWith('image/')) handleImgFile(f[0]);
    else toast('File harus berupa gambar (PNG/JPG/WebP)!', 'err');
});
imgFi.addEventListener('change', e => { if(e.target.files.length) handleImgFile(e.target.files[0]); });

function handleImgFile(f) {
    imgFile = f;
    const url = URL.createObjectURL(f);
    imgEl = new Image();
    imgEl.onload = () => {
        imgSrcW = imgEl.naturalWidth;
        imgSrcH = imgEl.naturalHeight;
        URL.revokeObjectURL(url);

        document.getElementById('img-fn2').textContent  = f.name;
        document.getElementById('img-sRes').textContent = `${imgSrcW} × ${imgSrcH}`;
        document.getElementById('img-sSiz').textContent = fSiz(f.size);

        document.getElementById('img-upload').style.display = 'none';
        document.getElementById('img-editor').style.display = 'block';

        buildScaleOpts();
        setTimeout(() => { initImgCompare(); aiInitUI(); }, 150);
    };
    imgEl.onerror = () => toast('Gagal memuat gambar!', 'err');
    imgEl.src = url;
}

// ═══════════════════════════════════════════════════════
// Scale Options Builder
// ═══════════════════════════════════════════════════════
function buildScaleOpts() {
    const c = document.getElementById('img-scale-opts');
    c.innerHTML = '';

    // Detect WebGL hardware limits
    const testC = document.createElement('canvas');
    const testG = testC.getContext('webgl');
    const maxTex  = testG ? testG.getParameter(testG.MAX_TEXTURE_SIZE)  : 4096;
    const maxVP   = testG ? testG.getParameter(testG.MAX_VIEWPORT_DIMS) : null;
    const maxSide = maxVP ? Math.min(maxTex, maxVP[0], maxVP[1]) : maxTex;

    const isPortrait = imgSrcH > imgSrcW;
    const srcLong    = Math.max(imgSrcW, imgSrcH);

    // Target resolutions by longest side
    const TARGETS = [
        { label: 'FHD', longSide: 1920, badge: '' },
        { label: '2K',  longSide: 2560, badge: '' },
        { label: '4K',  longSide: 3840, badge: 'HD' },
        { label: '8K',  longSide: 7680, badge: 'ULTRA' },
    ];

    TARGETS.forEach(t => {
        // Compute output dimensions maintaining source aspect ratio
        const outW = isPortrait
            ? Math.round(imgSrcW * (t.longSide / srcLong))
            : t.longSide;
        const outH = isPortrait
            ? t.longSide
            : Math.round(imgSrcH * (t.longSide / srcLong));

        const tooSmall = t.longSide <= srcLong;  // already at or beyond this res
        const tooLarge = outW > maxSide || outH > maxSide;

        const d = document.createElement('div');
        d.className = 'scale-opt' + (tooLarge || tooSmall ? ' dis' : '');
        d.dataset.targetW = outW;
        d.dataset.targetH = outH;
        d.dataset.label   = t.label;

        const badge = t.badge ? `<div class="extreme-badge">${t.badge}</div>` : '';
        const resHtml = tooLarge
            ? `<div class="sn-res" style="color:var(--red)">✕ Melebihi limit GPU</div>`
            : tooSmall
            ? `<div class="sn-res" style="opacity:.5">~ Sudah cukup besar</div>`
            : `<div class="sn-res">${outW}×${outH}</div>`;

        d.innerHTML = `<div class="sn-val">${t.label}</div>${resHtml}${badge}`;

        if(!tooLarge && !tooSmall) d.onclick = () => {
            c.querySelectorAll('.scale-opt').forEach(o => o.classList.remove('sel'));
            d.classList.add('sel');
            imgSelTargetW = outW;
            imgSelTargetH = outH;
            imgSelLabel   = t.label;
            updateScaleInfo();
        };
        c.appendChild(d);
    });

    // Auto-select: prefer 4K, then highest available
    const avail  = [...c.querySelectorAll('.scale-opt:not(.dis)')];
    const pref4K = avail.find(el => el.dataset.label === '4K');
    const autoSel = pref4K || avail[avail.length - 1] || avail[0];
    if(autoSel) autoSel.click();
}

function updateScaleInfo() {
    if(!imgSrcW || !imgSelTargetW) return;
    // Count passes: double from src until reaching target
    let pW = imgSrcW, pH = imgSrcH, passes = 0;
    while(pW < imgSelTargetW || pH < imgSelTargetH) {
        pW = Math.min(pW * 2, imgSelTargetW);
        pH = Math.min(pH * 2, imgSelTargetH);
        passes++;
    }
    const scaleX = (imgSelTargetW / imgSrcW).toFixed(1);
    document.getElementById('img-out-res').textContent = `${imgSelTargetW} × ${imgSelTargetH}`;
    document.getElementById('img-passes').textContent  = `${passes} GPU Pass · ${scaleX}×`;
}

// ═══════════════════════════════════════════════════════
// WebGL Init & Render (Image)
// ═══════════════════════════════════════════════════════
function initImgGL(canvas) {
    const c = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: true });
    if(!c) return null;

    const mkS = (type, src) => {
        const s = c.createShader(type);
        c.shaderSource(s, src); c.compileShader(s);
        if(!c.getShaderParameter(s, c.COMPILE_STATUS)) { console.error(c.getShaderInfoLog(s)); return null; }
        return s;
    };

    const vs = mkS(c.VERTEX_SHADER,   IMG_VERT);
    const fs = mkS(c.FRAGMENT_SHADER, IMG_FRAG);
    if(!vs || !fs) return null;

    const p = c.createProgram();
    c.attachShader(p, vs); c.attachShader(p, fs); c.linkProgram(p);
    if(!c.getProgramParameter(p, c.LINK_STATUS)) { console.error(c.getProgramInfoLog(p)); return null; }
    c.useProgram(p);

    const buf = c.createBuffer();
    c.bindBuffer(c.ARRAY_BUFFER, buf);
    c.bufferData(c.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), c.STATIC_DRAW);
    const ap = c.getAttribLocation(p, 'a_pos');
    c.enableVertexAttribArray(ap);
    c.vertexAttribPointer(ap, 2, c.FLOAT, false, 0, 0);

    const t = c.createTexture();
    c.bindTexture(c.TEXTURE_2D, t);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_S,     c.CLAMP_TO_EDGE);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_T,     c.CLAMP_TO_EDGE);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MIN_FILTER, c.LINEAR);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MAG_FILTER, c.LINEAR);
    c.pixelStorei(c.UNPACK_FLIP_Y_WEBGL, true);

    return {
        gl: c, tex: t,
        u: {
            img:   c.getUniformLocation(p, 'u_img'),
            ts:    c.getUniformLocation(p, 'u_ts'),
            isz:   c.getUniformLocation(p, 'u_isz'),
            sharp: c.getUniformLocation(p, 'u_sharp'),
            clar:  c.getUniformLocation(p, 'u_clar'),
            vib:   c.getUniformLocation(p, 'u_vib'),
            dens:  c.getUniformLocation(p, 'u_dens'),
            noise: c.getUniformLocation(p, 'u_noise'),
        }
    };
}

// ─── Fractal IFS WebGL Init ────────────────────────────────────────────────────
function initImgGLFractal(canvas) {
    const c = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: true });
    if(!c) return null;

    const mkS = (type, src) => {
        const s = c.createShader(type);
        c.shaderSource(s, src); c.compileShader(s);
        if(!c.getShaderParameter(s, c.COMPILE_STATUS)) {
            console.error('[FractalShader]', c.getShaderInfoLog(s)); return null;
        }
        return s;
    };

    const vs = mkS(c.VERTEX_SHADER,   IMG_VERT);
    const fs = mkS(c.FRAGMENT_SHADER, IMG_FRAG_FRACTAL);
    if(!vs || !fs) return null;

    const p = c.createProgram();
    c.attachShader(p, vs); c.attachShader(p, fs); c.linkProgram(p);
    if(!c.getProgramParameter(p, c.LINK_STATUS)) {
        console.error('[FractalProgram]', c.getProgramInfoLog(p)); return null;
    }
    c.useProgram(p);

    const buf = c.createBuffer();
    c.bindBuffer(c.ARRAY_BUFFER, buf);
    c.bufferData(c.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), c.STATIC_DRAW);
    const ap = c.getAttribLocation(p, 'a_pos');
    c.enableVertexAttribArray(ap);
    c.vertexAttribPointer(ap, 2, c.FLOAT, false, 0, 0);

    const t = c.createTexture();
    c.bindTexture(c.TEXTURE_2D, t);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_S,     c.CLAMP_TO_EDGE);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_WRAP_T,     c.CLAMP_TO_EDGE);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MIN_FILTER, c.LINEAR);
    c.texParameteri(c.TEXTURE_2D, c.TEXTURE_MAG_FILTER, c.LINEAR);
    c.pixelStorei(c.UNPACK_FLIP_Y_WEBGL, true);

    return {
        gl: c, tex: t,
        u: {
            img:        c.getUniformLocation(p, 'u_img'),
            ts:         c.getUniformLocation(p, 'u_ts'),
            isz:        c.getUniformLocation(p, 'u_isz'),
            sharp:      c.getUniformLocation(p, 'u_sharp'),
            clar:       c.getUniformLocation(p, 'u_clar'),
            vib:        c.getUniformLocation(p, 'u_vib'),
            dens:       c.getUniformLocation(p, 'u_dens'),
            noise:      c.getUniformLocation(p, 'u_noise'),
            frac_detail:c.getUniformLocation(p, 'u_frac_detail'),
        }
    };
}

// ─── Fractal IFS Render ───────────────────────────────────────────────────────
function renderImgGLFractal(ref, source, srcW, srcH, opts) {
    const g = ref.gl;
    g.viewport(0, 0, g.canvas.width, g.canvas.height);
    g.bindTexture(g.TEXTURE_2D, ref.tex);
    g.texImage2D(g.TEXTURE_2D, 0, g.RGBA, g.RGBA, g.UNSIGNED_BYTE, source);
    g.uniform1i(ref.u.img,          0);
    g.uniform2f(ref.u.ts,           1 / srcW,  1 / srcH);
    g.uniform2f(ref.u.isz,          srcW,       srcH);
    g.uniform1f(ref.u.sharp,        opts.sharp  || 0);
    g.uniform1f(ref.u.clar,         opts.clar   || 0);
    g.uniform1f(ref.u.vib,          opts.vib    || 0);
    g.uniform1f(ref.u.dens,         opts.dens   || 0);
    g.uniform1f(ref.u.noise,        opts.noise  || 0);
    g.uniform1f(ref.u.frac_detail,  opts.frac_detail !== undefined ? opts.frac_detail : 0.85);
    g.drawArrays(g.TRIANGLE_STRIP, 0, 4);
}

// ─── Engine Selector ──────────────────────────────────────────────────────────
function setImgEngine(engine) {
    imgEngine = engine; // 'bicubic' | 'fractal'
    // Update button states
    document.querySelectorAll('.eng-btn').forEach(b => b.classList.remove('sel'));
    const btn = document.getElementById(`eng-btn-${engine}`);
    if(btn) btn.classList.add('sel');
    // Update info text
    const info = document.getElementById('engine-info');
    if(info) {
        info.innerHTML = engine === 'fractal'
            ? '✦ <strong>Genuine Fractal IFS</strong> — self-similarity search + IFS transform. Terbaik untuk foto bertekstur, alam, dan arsitektur.'
            : '⚡ <strong>Bicubic Catmull-Rom</strong> — multi-pass GPU, HF synthesis. Terbaik untuk upscale cepat dan portrait.';
    }
    // Re-render preview with new engine
    renderImgPreview();
    // Refresh pass info label
    const pl = document.getElementById('img-ppl');
    if(pl) pl.textContent = engine === 'fractal' ? 'Fractal IFS' : 'Extreme';
}

function renderImgGL(ref, source, srcW, srcH, opts) {
    const g = ref.gl;
    g.viewport(0, 0, g.canvas.width, g.canvas.height);
    g.bindTexture(g.TEXTURE_2D, ref.tex);
    g.texImage2D(g.TEXTURE_2D, 0, g.RGBA, g.RGBA, g.UNSIGNED_BYTE, source);
    g.uniform1i(ref.u.img,   0);
    g.uniform2f(ref.u.ts,    1 / srcW,       1 / srcH);    // source texel size
    g.uniform2f(ref.u.isz,   srcW,            srcH);        // source dimensions (for bicubic)
    g.uniform1f(ref.u.sharp, opts.sharp || 0);
    g.uniform1f(ref.u.clar,  opts.clar  || 0);
    g.uniform1f(ref.u.vib,   opts.vib   || 0);
    g.uniform1f(ref.u.dens,  opts.dens  || 0);
    g.uniform1f(ref.u.noise, opts.noise || 0);
    g.drawArrays(g.TRIANGLE_STRIP, 0, 4);
}

// ═══════════════════════════════════════════════════════
// Comparison Preview
// ═══════════════════════════════════════════════════════
function initImgCompare() {
    const W      = imgCmpWrap.offsetWidth || 800;
    const aspect = imgSrcH / imgSrcW;
    const dW     = Math.min(W, Math.max(imgSrcW, 300));
    const dH     = Math.round(dW * aspect);

    // Size the wrapper
    imgCmpWrap.style.height = dH + 'px';

    // Set canvas internal resolution
    imgBeforeC.width  = dW; imgBeforeC.height = dH;
    imgAfterC.width   = dW; imgAfterC.height  = dH;

    // Draw original image on before canvas (2D)
    const ctx = imgBeforeC.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, dW, dH);

    // WebGL on after canvas — use selected engine
    imgCmpRef = imgEngine === 'fractal'
        ? initImgGLFractal(imgAfterC)
        : initImgGL(imgAfterC);
    if(imgCmpRef) imgCmpRef._isFractal = (imgEngine === 'fractal');
    if(!imgCmpRef) { toast('WebGL tidak tersedia untuk preview!', 'err'); return; }

    setImgDivider(50);
    renderImgPreview();
    initImgDrag();
}

function renderImgPreview() {
    if(!imgEl) return;
    const gv = id => +document.getElementById(id).value / 100;
    const opts = {
        sharp: gv('img-shS'),
        clar:  gv('img-clS'),
        vib:   gv('img-viS'),
        dens:  gv('img-dnS'),
        noise: gv('img-noS'),
        frac_detail: 0.85,
    };
    if(imgEngine === 'fractal') {
        // Re-init fractal GL on after-canvas if needed
        if(!imgCmpRef || !imgCmpRef._isFractal) {
            imgCmpRef = initImgGLFractal(imgAfterC);
            if(imgCmpRef) imgCmpRef._isFractal = true;
        }
        if(imgCmpRef) renderImgGLFractal(imgCmpRef, imgEl, imgSrcW, imgSrcH, opts);
    } else {
        if(!imgCmpRef || imgCmpRef._isFractal) {
            imgCmpRef = initImgGL(imgAfterC);
            if(imgCmpRef) imgCmpRef._isFractal = false;
        }
        if(imgCmpRef) renderImgGL(imgCmpRef, imgEl, imgSrcW, imgSrcH, opts);
    }
}

function imgUpC() {
    ['sh','cl','vi','dn','no'].forEach(k => {
        const sv = document.getElementById(`img-${k}S`);
        const vv = document.getElementById(`img-${k}V`);
        if(sv && vv) vv.textContent = sv.value + '%';
    });
    renderImgPreview();
}

// ═══════════════════════════════════════════════════════
// Comparison Divider (Drag to compare)
// ═══════════════════════════════════════════════════════
function setImgDivider(pct) {
    imgDivPct = Math.max(0, Math.min(100, pct));
    imgDivider.style.left    = imgDivPct + '%';
    // Clip after-canvas from the LEFT so right portion shows over before-canvas
    imgAfterC.style.clipPath = `inset(0 0 0 ${imgDivPct}%)`;

    const bl = document.querySelector('.img-lbl-before');
    const al = document.querySelector('.img-lbl-after');
    if(bl) bl.style.opacity = imgDivPct > 8  ? '1' : '0';
    if(al) al.style.opacity = imgDivPct < 92 ? '1' : '0';
}

function initImgDrag() {
    const onMove = clientX => {
        if(!imgIsDrag) return;
        const r = imgCmpWrap.getBoundingClientRect();
        setImgDivider((clientX - r.left) / r.width * 100);
    };

    imgDivider.addEventListener('mousedown',  ()  => imgIsDrag = true);
    imgCmpWrap.addEventListener('mousedown',  e   => { imgIsDrag = true; onMove(e.clientX); });
    document.addEventListener('mousemove',   e   => onMove(e.clientX));
    document.addEventListener('mouseup',     ()  => imgIsDrag = false);

    imgDivider.addEventListener('touchstart', e => { imgIsDrag = true; e.preventDefault(); }, { passive: false });
    imgCmpWrap.addEventListener('touchmove',  e => { onMove(e.touches[0].clientX); e.preventDefault(); }, { passive: false });
    document.addEventListener('touchend',    ()  => imgIsDrag = false);
}

// ═══════════════════════════════════════════════════════
// Format Selection
// ═══════════════════════════════════════════════════════
function selImgF(el) {
    document.querySelectorAll('[data-imgfmt]').forEach(f => f.classList.remove('sel'));
    el.classList.add('sel');
    imgSelFmt = el.dataset.imgfmt;
}

// ═══════════════════════════════════════════════════════
// Multi-Pass Processing
// ═══════════════════════════════════════════════════════
async function processImg() {
    if(!imgEl || imgIsProc) return;
    imgIsProc = true;

    document.getElementById('img-editor').style.display   = 'none';
    document.getElementById('img-proc').style.display     = 'block';
    document.getElementById('img-proc-btn').disabled      = true;

    const gv = id => +document.getElementById(id).value / 100;
    const sh = gv('img-shS'), cl = gv('img-clS'), vi = gv('img-viS');
    const dn = gv('img-dnS'), no = gv('img-noS');

    // Compute pass sequence: double from source toward target
    // Each pass doubles, last pass may land exactly on target (not necessarily 2×)
    const passSeq = [];
    { let pW = imgSrcW, pH = imgSrcH;
      while(pW < imgSelTargetW || pH < imgSelTargetH) {
          const nW = Math.min(pW * 2, imgSelTargetW);
          const nH = Math.min(pH * 2, imgSelTargetH);
          passSeq.push({ w: nW, h: nH });
          pW = nW; pH = nH;
      }
      if(passSeq.length === 0) passSeq.push({ w: imgSelTargetW, h: imgSelTargetH });
    }
    const totalPasses = passSeq.length;

    try {
        let source = imgEl, currW = imgSrcW, currH = imgSrcH;
        let lastRef = null;
        clearLog('img-log');
        addLog('img-log', `🖼️ Gambar: ${imgSrcW}×${imgSrcH} → ${imgSelTargetW}×${imgSelTargetH} (${totalPasses} pass)`, 'gpu');
        addLog('img-log', `⚙️ Sharp: ${Math.round(sh*100)}% | Clarity: ${Math.round(cl*100)}% | Vibrance: ${Math.round(vi*100)}% | Denoise: ${Math.round(no*100)}%`, 'info');

        for(let pass = 0; pass < totalPasses; pass++) {
            const nextW = passSeq[pass].w, nextH = passSeq[pass].h;
            const pct = (pass / totalPasses) * 100;

            // Update progress UI
            document.getElementById('img-pP').textContent = Math.round(pct) + '%';
            document.getElementById('img-pF').style.width = pct + '%';
            document.getElementById('img-pS').innerHTML   =
                `<strong>Pass ${pass+1}/${totalPasses}</strong> — ${currW}×${currH} → <span style="color:var(--accent2)">${nextW}×${nextH}</span>`;
            addLog('img-log', `🎯 Pass ${pass+1}/${totalPasses}: ${currW}×${currH} → ${nextW}×${nextH}`, 'gpu');
            await sleep(20); // yield to browser for UI repaint

            const outCanvas = document.createElement('canvas');
            outCanvas.width  = nextW;
            outCanvas.height = nextH;

            // Init correct GL engine for this pass
            const ref = imgEngine === 'fractal'
                ? initImgGLFractal(outCanvas)
                : initImgGL(outCanvas);
            if(!ref) throw new Error(`WebGL gagal pada pass ${pass + 1}`);
            lastRef = ref;

            // Verify the viewport wasn't silently clamped by the driver
            ref.gl.viewport(0, 0, outCanvas.width, outCanvas.height);
            const vp = ref.gl.getParameter(ref.gl.VIEWPORT);
            if(vp[2] !== nextW || vp[3] !== nextH) {
                throw new Error(
                    `GPU viewport dibatasi jadi ${vp[2]}×${vp[3]}, ` +
                    `dibutuhkan ${nextW}×${nextH}. Coba skala lebih kecil.`
                );
            }

            // Enhancement strength decays per pass to avoid over-sharpening
            const decay = Math.pow(0.60, pass); // 1.0 → 0.60 → 0.36 → 0.22

            // Fractal detail fades slightly each pass to avoid over-synthesis
            const fracDetail = Math.pow(0.75, pass) * 0.85;

            const passOpts = {
                sharp: sh    * decay,
                clar:  cl    * decay,
                vib:   pass === 0 ? vi : 0,   // vibrance only on first pass
                dens:  dn,                      // density applied every pass
                noise: pass === 0 ? no : 0,   // denoise only on first pass
                frac_detail: fracDetail,
            };

            if(imgEngine === 'fractal') {
                renderImgGLFractal(ref, source, currW, currH, passOpts);
            } else {
                renderImgGL(ref, source, currW, currH, passOpts);
            }

            // ⚠️  CRITICAL: wait for GPU to finish BEFORE the canvas is used as
            // the next pass's texture source or handed to toBlob().
            // Without this, the browser may read an incomplete framebuffer,
            // producing a cropped/partial PNG.
            ref.gl.finish();

            source = outCanvas;
            currW  = nextW;
            currH  = nextH;
        }

        // Finalize — flush GPU before anything reads the source canvas
        if(lastRef) { lastRef.gl.finish(); }
        await sleep(80);

        // ⚠️  CRITICAL anti-crop: copy WebGL result to a 2D canvas RIGHT NOW,
        // before any async AI call. WebGL drawing buffers can be silently cleared
        // by the browser during long API calls → causes crop on fallback.
        {
            const safeCanvas = document.createElement('canvas');
            safeCanvas.width  = currW;
            safeCanvas.height = currH;
            safeCanvas.getContext('2d').drawImage(source, 0, 0, currW, currH);
            source = safeCanvas; // from now on, always read from this safe 2D copy
        }

        document.getElementById('img-pP').textContent = '99%';
        document.getElementById('img-pF').style.width = '99%';
        document.getElementById('img-pS').innerHTML   = '<strong>Finalisasi &amp; Encoding...</strong>';

        // ── AI Generate Pass (optional) ──────────────────────────────────
        aiResultWasUsed = false;
        if(aiEnabled) {
            // Resolve API key from the active provider's input
            const aiKey = aiProvider === 'huggingface'
                ? document.getElementById('ai-hf-key')?.value?.trim()
                : aiProvider === 'replicate'
                ? document.getElementById('ai-rep-key')?.value?.trim()
                : aiProvider === 'realesrgan_local'
                ? 'local'   // no real key needed
                : /* gemini */
                  document.getElementById('ai-gem-key')?.value?.trim()
                  || (typeof DEFAULT_GEMINI_KEY !== 'undefined' ? DEFAULT_GEMINI_KEY : '');

            if(!aiKey && aiProvider !== 'realesrgan_local') {
                toast('AI aktif tapi API key kosong — lewati AI pass.', 'err');
            } else {
                const modelId = aiProvider === 'huggingface'
                    ? document.getElementById('ai-hf-model')?.value
                    : aiProvider === 'replicate'
                    ? document.getElementById('ai-rep-model')?.value
                    : aiProvider === 'realesrgan_local'
                    ? document.getElementById('ai-local-model')?.value
                    : document.getElementById('ai-gem-model')?.value;

                const strength = (+document.getElementById('ai-strength-s')?.value || 85) / 100;

                // Show AI progress overlay
                document.getElementById('img-pS').innerHTML = '<strong>✦ AI Generate Pass...</strong>';
                document.getElementById('ai-prog-overlay').classList.add('vis');

                // ⚡ Capture sumber sebagai base64 SINKRON sebelum operasi async apapun.
                // Canvas WebGL bersifat volatile — buffer-nya di-clear browser saat async.
                // toDataURL() membaca piksel SEKARANG sebelum browser menghapusnya.
                let srcB64 = null;
                try {
                    srcB64 = source.toDataURL('image/jpeg', 0.92);
                } catch(captureErr) {
                    toast('Gagal capture gambar: ' + captureErr.message, 'err');
                }

                if (srcB64) {
                    try {
                        const aiCanvas = await aiUpscale(srcB64, currW, currH, {
                            provider:   aiProvider,
                            apiKey:     aiKey,
                            modelId,
                            strength,
                            onProgress: (msg, pct) => {
                                document.getElementById('ai-prog-msg').textContent = msg;
                                document.getElementById('ai-prog-bar').style.width = (pct || 0) + '%';
                                document.getElementById('ai-prog-bar').style.setProperty('--pct', (pct || 0) + '%');
                                addLog('img-log', `🤖 ${msg}`, 'ai');
                            },
                        });
                        // Replace source with AI-enhanced canvas (dimensions may differ)
                        source = aiCanvas;
                        currW  = aiCanvas.width;
                        currH  = aiCanvas.height;
                        aiResultWasUsed = true;
                        addLog('img-log', `✨ AI selesai: ${currW}×${currH}`, 'ok');
                    } catch(aiErr) {
                        console.error('[AI]', aiErr);
                        addLog('img-log', `⚠️ AI error: ${aiErr.message}`, 'err');
                        toast('AI error: ' + aiErr.message + ' — menggunakan hasil WebGL.', 'err');
                    } finally {
                        document.getElementById('ai-prog-overlay').classList.remove('vis');
                    }
                } else {
                    document.getElementById('ai-prog-overlay').classList.remove('vis');
                }
            } // end if(!aiKey) else
        } // end if(aiEnabled)

        // ── Export to blob ────────────────────────────────────────────────
        // Copy source into a fresh 2D canvas before toBlob().
        // drawImage() works on HTMLImageElement, WebGL canvas, and 2D canvas
        // so this is safe regardless of what source is.
        const mime = imgSelFmt === 'webp' ? 'image/webp' : 'image/png';
        const exportCanvas = document.createElement('canvas');
        exportCanvas.width  = currW;
        exportCanvas.height = currH;
        exportCanvas.getContext('2d').drawImage(source, 0, 0, currW, currH);

        const blob = await new Promise(res => exportCanvas.toBlob(res, mime, 0.95));
        if(!blob) throw new Error('Ekspor gambar gagal');

        if(imgResultUrl) URL.revokeObjectURL(imgResultUrl);
        imgResultUrl = URL.createObjectURL(blob);

        document.getElementById('img-pP').textContent = '100%';
        document.getElementById('img-pF').style.width = '100%';
        await sleep(250);

        addLog('img-log', `✅ Selesai! ${imgSrcW}×${imgSrcH} → ${currW}×${currH} | ${fSiz(blob.size)}${aiResultWasUsed ? ' + AI' : ''}`, 'ok');
        // Build enhancement label
        const scaleX = (currW / imgSrcW).toFixed(1);
        const engineLabel = imgEngine === 'fractal' ? 'Fractal IFS' : 'GPU Enhanced';
        const enhLabel = aiResultWasUsed
            ? `${imgSelLabel} ${engineLabel} + <span class="ai-enhanced-badge">🍌 AI Auto Settings</span>`
            : `${imgSelLabel} · ${scaleX}× ${engineLabel}`;

        // Show result
        document.getElementById('img-proc').style.display     = 'none';
        document.getElementById('img-result').style.display   = 'block';
        document.getElementById('img-res-before').textContent = `${imgSrcW}×${imgSrcH}`;
        document.getElementById('img-res-after').textContent  = `${currW}×${currH}`;
        document.getElementById('img-res-size').textContent   = fSiz(blob.size);
        document.getElementById('img-res-scale').innerHTML    = enhLabel;
        document.getElementById('img-result-preview').src     = imgResultUrl;

        imgIsProc = false;
        toast(`Selesai! Upscale ke ${imgSelLabel} (${currW}×${currH})${aiResultWasUsed ? ' + AI' : ''} berhasil.`);

    } catch(err) {
        console.error(err);
        toast('Error: ' + err.message, 'err');
        imgIsProc = false;
        document.getElementById('img-proc').style.display   = 'none';
        document.getElementById('img-editor').style.display = 'block';
        document.getElementById('img-proc-btn').disabled    = false;
    }
}

// ═══════════════════════════════════════════════════════
// Download
// ═══════════════════════════════════════════════════════
function dlImg() {
    if(!imgResultUrl) return;
    const a  = document.createElement('a');
    a.href   = imgResultUrl;
    const bn  = imgFile ? imgFile.name.replace(/\.[^.]+$/, '') : 'image';
    const sfx = aiResultWasUsed ? `${imgSelLabel}_ai` : `${imgSelLabel}`;
    a.download = `${bn}_${sfx}.${imgSelFmt}`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    toast('Download gambar dimulai!');
}

// ═══════════════════════════════════════════════════════
// Back to Editor / Reset
// ═══════════════════════════════════════════════════════
function backToEditor() {
    document.getElementById('img-result').style.display = 'none';
    document.getElementById('img-editor').style.display = 'block';
    document.getElementById('img-proc-btn').disabled    = false;
}

function resetImg() {
    if(imgResultUrl) URL.revokeObjectURL(imgResultUrl);
    imgFile = null; imgEl = null; imgSrcW = 0; imgSrcH = 0;
    imgCmpRef = null; imgResultUrl = null; imgIsProc = false;
    aiResultWasUsed = false;
    imgFi.value = '';
    document.getElementById('img-upload').style.display = 'block';
    document.getElementById('img-editor').style.display = 'none';
    document.getElementById('img-proc').style.display   = 'none';
    document.getElementById('img-result').style.display = 'none';
    document.getElementById('img-proc-btn').disabled    = false;
    document.getElementById('ai-prog-overlay').classList.remove('vis');
}

// ═══════════════════════════════════════════════════════
// Tab Switching
// ═══════════════════════════════════════════════════════
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tab}"]`).classList.add('active');
    document.getElementById('video-panel').style.display = tab === 'video' ? 'block' : 'none';
    document.getElementById('img-panel').style.display   = tab === 'image' ? 'block' : 'none';
}

// ═══════════════════════════════════════════════════════
// AI UI Controllers
// ═══════════════════════════════════════════════════════

/** Populate model selects and restore saved keys on first load */
function aiInitUI() {
    if(typeof aiPopulateModels === 'function') {
        aiPopulateModels('huggingface',      document.getElementById('ai-hf-model'));
        aiPopulateModels('replicate',        document.getElementById('ai-rep-model'));
        aiPopulateModels('gemini',           document.getElementById('ai-gem-model'));
        aiPopulateModels('realesrgan_local', document.getElementById('ai-local-model'));
    }
    if(typeof aiRestoreKeys === 'function') aiRestoreKeys();
    const gemKey = typeof loadKey === 'function' ? loadKey('gemini') : '';
    const vGem = document.getElementById('vid-gem-key');
    if(vGem && gemKey) vGem.value = gemKey;
    onAIStrength(85);
    // Check Real-ESRGAN server status silently on init
    checkESRGANServer();
}

/** Toggle AI section open/closed */
function toggleAI() {
    const chk = document.getElementById('ai-toggle-chk');
    chk.checked = !chk.checked;
    onAIToggle(chk.checked);
}

/** Called when toggle checkbox changes */
function onAIToggle(checked) {
    aiEnabled = checked;
    const sec = document.getElementById('ai-section');
    sec.classList.toggle('active', checked);
    // Update button label
    const lbl = document.getElementById('proc-btn-label');
    if(lbl) lbl.textContent = checked ? '✦ Proses + AI' : 'Proses Extreme';
}

/** Switch between AI provider tabs */
function switchAIProvider(prov) {
    aiProvider = prov;
    const tabs = [
        { id: 'hf',    prov: 'huggingface'      },
        { id: 'rep',   prov: 'replicate'         },
        { id: 'gem',   prov: 'gemini'            },
        { id: 'local', prov: 'realesrgan_local'  },
    ];
    tabs.forEach(t => {
        document.getElementById(`ai-prov-${t.id}`)?.classList.toggle('sel',    prov === t.prov);
        document.getElementById(`ai-panel-${t.id}`)?.classList.toggle('active', prov === t.prov);
    });
    // Auto-check server status when switching to local tab
    if(prov === 'realesrgan_local') checkESRGANServer();
}

/** Ping local Real-ESRGAN server and update status indicator */
async function checkESRGANServer() {
    const dot = document.getElementById('esrgan-dot');
    const txt = document.getElementById('esrgan-status-txt');
    if(!dot || !txt) return;
    dot.style.background = '#f59e0b';
    dot.style.boxShadow  = '0 0 6px #f59e0b';
    txt.textContent = 'Mengecek server...';
    try {
        const r = await fetch('/api/ping', { signal: AbortSignal.timeout(6000) });
        if(r.ok) {
            const d = await r.json();
            dot.style.background = '#22c55e';
            dot.style.boxShadow  = '0 0 6px #22c55e';
            txt.textContent = `✅ Server aktif — device: ${d.device}`;
        } else { throw new Error('not ok'); }
    } catch(_) {
        dot.style.background = '#ef4444';
        dot.style.boxShadow  = 'none';
        txt.textContent = '❌ Server tidak aktif — tunggu sebentar atau restart Mac';
    }
}

/** Strength slider update */
function onAIStrength(val) {
    const pct = val + '%';
    document.getElementById('ai-strength-v').textContent = pct;
    const sl = document.getElementById('ai-strength-s');
    if(sl) sl.style.setProperty('--pct', pct);
}

/** Show/hide API key (eye button) */
function toggleKeyVisibility(inputId, btn) {
    const el = document.getElementById(inputId);
    if(!el) return;
    el.type = el.type === 'password' ? 'text' : 'password';
    // Swap icon
    btn.innerHTML = el.type === 'text'
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
              <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
              <line x1="1" y1="1" x2="23" y2="23"/>
           </svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
              <circle cx="12" cy="12" r="3"/>
           </svg>`;
}

// Fallback for AI canvas (2D) toBlob if method is missing
function canvasToBlobFallback(canvas, mime) {
    return new Promise(res => canvas.toBlob(res, mime, 0.95));
}

// ═══════════════════════════════════════════════════════
// Gemini Video AI Analysis
// ═══════════════════════════════════════════════════════

let _geminiVideoParams = null; // store last Gemini recommendation

function toggleVidAI() {
    const chk = document.getElementById('vid-ai-chk');
    chk.checked = !chk.checked;
    onVidAIToggle(chk.checked);
}

function onVidAIToggle(checked) {
    const sec = document.getElementById('vid-ai-section');
    sec.classList.toggle('active', checked);
    // Sync Gemini key from saved store
    if(checked && typeof loadKey === 'function') {
        const saved = loadKey('gemini');
        const vGem = document.getElementById('vid-gem-key');
        if(vGem && !vGem.value && saved) vGem.value = saved;
    }
}

/**
 * Captures the first video frame, sends to Gemini,
 * gets back optimal shader params, and shows them.
 */
async function geminiAnalyzeVideo() {
    const apiKey = document.getElementById('vid-gem-key')?.value?.trim();
    if(!apiKey) { toast('Masukkan Gemini API key terlebih dahulu.', 'err'); return; }

    // Need a video loaded
    const pv = document.getElementById('pv');
    if(!pv || !pv.videoWidth) { toast('Upload video terlebih dahulu.', 'err'); return; }

    const btn = document.getElementById('vid-ai-btn');
    if(btn) { btn.disabled = true; btn.textContent = 'Gemini sedang menganalisa...'; }
    document.getElementById('vid-ai-result').style.display = 'none';

    try {
        // Capture current frame to an offscreen canvas
        const fc = document.createElement('canvas');
        fc.width  = pv.videoWidth;
        fc.height = pv.videoHeight;
        fc.getContext('2d').drawImage(pv, 0, 0);

        if(typeof geminiVideoAnalyze !== 'function') throw new Error('ai-engine.js tidak termuat.');
        _geminiVideoParams = await geminiVideoAnalyze(fc, apiKey);

        // Display result
        const p = _geminiVideoParams;
        const sceneLabel = {
            outdoor: '🌳 Outdoor / Alam',
            indoor:  '🏠 Indoor / Ruangan',
            portrait: '👤 Portrait / Wajah',
            nighttime: '🌙 Malam / Low-Light',
            animation: '🎨 Animasi / Cartoon',
            document: '📝 Teks / Dokumen',
        }[p.sceneType] || '🎥 ' + p.sceneType;

        document.getElementById('vid-ai-scene').innerHTML  =
            `<strong>Scene:</strong> ${sceneLabel} &nbsp;|&nbsp; ` +
            `Sharpen: <strong>${Math.round(p.sharp*100)}%</strong> &nbsp;` +
            `Clarity: <strong>${Math.round(p.clar*100)}%</strong> &nbsp;` +
            `Vibrance: <strong>${Math.round(p.vib*100)}%</strong> &nbsp;` +
            `Density: <strong>${p.dens.toFixed(1)}</strong>`;
        document.getElementById('vid-ai-reason').textContent = p.reasoning;
        document.getElementById('vid-ai-result').style.display = 'block';

        toast('✨ Gemini selesai menganalisa!', 'ok');
    } catch(err) {
        console.error('[GeminiVideo]', err);
        toast('Gemini error: ' + err.message, 'err');
    } finally {
        if(btn) {
            btn.disabled = false;
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Analisa dengan Gemini AI';
        }
    }
}

/**
 * Applies last Gemini recommendation to the video enhancement sliders.
 */
function applyGeminiParams() {
    if(!_geminiVideoParams) { toast('Analisa Gemini dulu!', 'err'); return; }
    const p = _geminiVideoParams;

    // Map to slider ranges:
    // shS: 0-200 (unsharp mask),  clS: 0-100 (clarity),
    // viS: 0-100 (vibrance),      dnS: 0-200 (pixel density)
    const setSlider = (id, val, labelId, suffix) => {
        const sl = document.getElementById(id);
        const lb = document.getElementById(labelId);
        if(sl) { sl.value = Math.round(val); }
        if(lb) lb.textContent = Math.round(val) + (suffix||'');
    };

    setSlider('shS', p.sharp * 200, 'shV', '%');
    setSlider('clS', p.clar  * 100, 'clV', '%');
    setSlider('viS', p.vib   * 100, 'viV', '%');
    setSlider('dnS', p.dens  * 100, 'dnV', '%');

    // Trigger preview refresh
    if(typeof renderCmp === 'function') renderCmp();
    if(typeof upC === 'function') upC();

    toast('✓ Setting Gemini diterapkan ke slider!', 'ok');
}
