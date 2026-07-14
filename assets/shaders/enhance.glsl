/* ══════════════════════════════════════════════════
   VideoScaler Pro — WebGL Shader Sources
   ══════════════════════════════════════════════════ */

/* Vertex Shader */
attribute vec2 a_pos;
varying vec2 v_uv;
void main(){
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}

/* ── Fragment Shader (paste this as FRAG template literal) ── */

precision highp float;
uniform sampler2D u_img;
uniform vec2 u_ts;
uniform vec2 u_isz;
uniform float u_sharp, u_clar, u_vib, u_cmp, u_dens;
varying vec2 v_uv;

/* Catmull-Rom cubic kernel */
float cubic(float x){
    float ax = abs(x);
    if(ax < 1.0) return  1.5*ax*ax*ax - 2.5*ax*ax + 1.0;
    if(ax < 2.0) return -0.5*ax*ax*ax + 2.5*ax*ax - 4.0*ax + 2.0;
    return 0.0;
}

/* Bicubic 4×4 tap sampler (Catmull-Rom) */
vec4 bicubic(vec2 uv){
    vec2 p  = uv * u_isz;
    vec2 f  = fract(p);
    p = floor(p);
    vec4  col = vec4(0.0);
    float tw  = 0.0;
    for(int y = -1; y <= 2; y++)
    for(int x = -1; x <= 2; x++){
        float wx = cubic(float(x) - f.x);
        float wy = cubic(float(y) - f.y);
        float w  = wx * wy;
        col += texture2D(u_img, (p + vec2(float(x), float(y))) / u_isz) * w;
        tw  += w;
    }
    return col / max(tw, 0.001);
}

/* Main enhancement pass */
vec4 enhance(vec2 uv, vec2 ts){
    /* 1. Blend bilinear → bicubic based on density (0..1 maps to 0%..50% of slider) */
    vec4  cBi = texture2D(u_img, uv);
    vec4  cBc = bicubic(uv);
    float db  = clamp(u_dens * 2.0, 0.0, 1.0);
    vec4  c   = mix(cBi, cBc, db);

    /* 2. Gaussian blur for Unsharp Mask */
    vec4  bl = vec4(0.0);
    float tw = 0.0;
    for(int y = -2; y <= 2; y++)
    for(int x = -2; x <= 2; x++){
        float d = float(x*x + y*y);
        float w = exp(-d * 0.35);
        bl += texture2D(u_img, uv + vec2(float(x), float(y)) * ts) * w;
        tw += w;
    }
    bl /= tw;

    /* 3. Unsharp Mask + Clarity */
    vec3 d2 = c.rgb - bl.rgb;
    vec3 r  = c.rgb + d2 * u_sharp;
    r += d2 * u_clar * 0.55;

    /* 4. Laplacian edge enhancement (clarity) */
    vec3 e = texture2D(u_img, uv + vec2( 0.0,  -ts.y)).rgb
           + texture2D(u_img, uv + vec2( 0.0,   ts.y)).rgb
           + texture2D(u_img, uv + vec2(-ts.x,  0.0 )).rgb
           + texture2D(u_img, uv + vec2( ts.x,  0.0 )).rgb
           - 4.0 * c.rgb;
    r += e * u_clar * 0.28;

    /* 5. High-frequency sub-pixel detail synthesis (density > 50%) */
    if(u_dens > 0.5){
        vec3 ne = texture2D(u_img, uv + vec2( ts.x, -ts.y) * 0.5).rgb;
        vec3 nw = texture2D(u_img, uv + vec2(-ts.x, -ts.y) * 0.5).rgb;
        vec3 se = texture2D(u_img, uv + vec2( ts.x,  ts.y) * 0.5).rgb;
        vec3 sw = texture2D(u_img, uv + vec2(-ts.x,  ts.y) * 0.5).rgb;
        vec3 hf = c.rgb - (ne + nw + se + sw) * 0.25;
        float xd = clamp((u_dens - 0.5) * 2.0, 0.0, 1.0);
        r += hf * xd * 0.65;
    }

    /* 6. Mild contrast boost */
    r = (r - 0.5) * 1.06 + 0.5;

    /* 7. Vibrance (selective saturation) */
    float lm  = dot(r, vec3(.2126, .7152, .0722));
    vec3  gr  = vec3(lm);
    float sat = length(r - gr);
    r = mix(gr, r, 1.0 + u_vib * (1.0 - min(sat * 4.0, 1.0)));

    return vec4(clamp(r, 0.0, 1.0), c.a);
}

void main(){
    vec2 uv = v_uv;

    /* Split-screen comparison mode */
    if(u_cmp > 0.5){
        if(uv.x < 0.497){
            gl_FragColor = texture2D(u_img, vec2(uv.x / 0.497, uv.y));
            return;
        } else if(uv.x > 0.503){
            uv = vec2((uv.x - 0.503) / 0.497, uv.y);
        } else {
            gl_FragColor = vec4(1.0);   /* divider line */
            return;
        }
    }

    gl_FragColor = enhance(uv, u_ts);
}
