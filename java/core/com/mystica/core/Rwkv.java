package com.mystica.core;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * RWKV-5.2 (x051a) 11M per-token recurrent engine.
 * Exact port of the verified Python Rwkv51 (app/recurrence parity ~1e-5 vs chunked forward).
 */
public final class Rwkv {
    public final int vocab;
    public final int nLayer;
    public final int nEmb;
    public final int nHead;
    public final int headSize;
    public final int blockSize;
    public final int padId, unkId, bosId, eosId;
    public final Tokenizer tok;

    private final float[] wte;          // V*C
    private final float[] lnF;          // C

    // per layer
    private final float[][] wln1, wln2;       // L*C
    private final float[][] tlma;             // L*4*C  order k,v,r,g
    private final float[][] wtr;                // L*C*C  receptance
    private final float[][] wtk;                // L*C*C  key
    private final float[][] wtv;                // L*C*C  value
    private final float[][] wtg;                // L*C*C  gate
    private final float[][] wto;                // L*C*C  output
    private final float[][] wlnxW, wlnxB;       // L*C
    private final float[] decay;               // L*H  (precomputed exp(-exp(min(x,20))))
    private final float[] faaa;                // L*H  raw u
    private final float[][] clma;              // L*2*C  order k,r
    private final float[][] cmk1, cmk2, cmr;   // (3C*C), (C*3C), (C*C)

    // recurrence state
    private float[] prevLn1;                    // L*C
    private float[] prevLn2;                    // L*C
    private final float[] sd;                   // L*H*N*N

    private Rwkv(int vocab, int nLayer, int nEmb, int nHead, int blockSize,
                 int padId, int unkId, int bosId, int eosId, Tokenizer tok,
                 Map<String, float[]> w) {
        this.vocab = vocab;
        this.nLayer = nLayer;
        this.nEmb = nEmb;
        this.nHead = nHead;
        this.headSize = nEmb / nHead;
        this.blockSize = blockSize;
        this.padId = padId;
        this.unkId = unkId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.tok = tok;

        int C = nEmb, L = nLayer, H = nHead, N = C / H;

        wte = w.get("lm_head.weight");
        lnF = w.get("transformer.ln_f.weight");
        wln1 = new float[L][];
        wln2 = new float[L][];
        tlma = new float[L][4 * C];
        wtr = new float[L][C * C];
        wtk = new float[L][C * C];
        wtv = new float[L][C * C];
        wtg = new float[L][C * C];
        wto = new float[L][C * C];
        wlnxW = new float[L][];
        wlnxB = new float[L][];
        decay = new float[L * H];
        faaa = new float[L * H];
        clma = new float[L][2 * C];
        cmk1 = new float[L][3 * C * C];
        cmk2 = new float[L][C * 3 * C];
        cmr = new float[L][C * C];

        for (int l = 0; l < L; l++) {
            wln1[l] = w.get("transformer.h." + l + ".ln_1.weight");
            wln2[l] = w.get("transformer.h." + l + ".ln_2.weight");
            float[] mk = w.get("transformer.h." + l + ".tmix.time_maa_k");
            float[] mv = w.get("transformer.h." + l + ".tmix.time_maa_v");
            float[] mr = w.get("transformer.h." + l + ".tmix.time_maa_r");
            float[] mg = w.get("transformer.h." + l + ".tmix.time_maa_g");
            for (int c = 0; c < C; c++) {
                tlma[l][c] = mk[c];
                tlma[l][C + c] = mv[c];
                tlma[l][2 * C + c] = mr[c];
                tlma[l][3 * C + c] = mg[c];
            }
            wtr[l] = w.get("transformer.h." + l + ".tmix.receptance.weight");
            wtk[l] = w.get("transformer.h." + l + ".tmix.key.weight");
            wtv[l] = w.get("transformer.h." + l + ".tmix.value.weight");
            wtg[l] = w.get("transformer.h." + l + ".tmix.gate.weight");
            wto[l] = w.get("transformer.h." + l + ".tmix.output.weight");
            wlnxW[l] = w.get("transformer.h." + l + ".tmix.ln_x.weight");
            wlnxB[l] = w.get("transformer.h." + l + ".tmix.ln_x.bias");
            float[] td = w.get("transformer.h." + l + ".tmix.time_decay");
            float[] tu = w.get("transformer.h." + l + ".tmix.time_faaaa");
            for (int h = 0; h < H; h++) {
                float d = td[h];
                decay[l * H + h] = (float) Math.exp(-Math.exp(Math.min(d, 20f)));
                faaa[l * H + h] = tu[h];
            }
            clma[l] = new float[2 * C];
            {
                float[] ck = w.get("transformer.h." + l + ".cmix.time_maa_k");
                float[] cr = w.get("transformer.h." + l + ".cmix.time_maa_r");
                for (int c = 0; c < C; c++) {
                    clma[l][c] = ck[c];
                    clma[l][C + c] = cr[c];
                }
            }
            cmk1[l] = w.get("transformer.h." + l + ".cmix.key.weight");
            cmk2[l] = w.get("transformer.h." + l + ".cmix.value.weight");
            cmr[l] = w.get("transformer.h." + l + ".cmix.receptance.weight");
        }

        prevLn1 = new float[L * C];
        prevLn2 = new float[L * C];
        sd = new float[L * H * N * N];
    }

    public static Rwkv load(InputStream is) throws IOException {
        BinIO r = new BinIO(is);
        String magic;
        {
            byte[] m = r.readBytes(5);
            magic = new String(m, java.nio.charset.StandardCharsets.US_ASCII);
        }
        BinIO.expectString(magic, "RWKV5", "magic");
        int ver = r.readU8();
        BinIO.expect(ver, 1, "version");
        int vocab = r.readU32();
        int L = r.readU32(), C = r.readU32(), H = r.readU32(), block = r.readU32();
        int padId = r.readU32(), unkId = r.readU32(), bosId = r.readU32(), eosId = r.readU32();
        BinIO.expect(C % H, 0, "n_embd % n_head");
        int nt = r.readU32();
        Map<String, float[]> w = new HashMap<>();
        for (int i = 0; i < nt; i++) {
            int nl = r.readU16();
            String name = new String(r.readBytes(nl), java.nio.charset.StandardCharsets.UTF_8);
            int numel = r.readU32();
            boolean fp32 = name.endsWith("time_decay") || name.endsWith("time_faaaa");
            float[] arr = new float[numel];
            if (fp32) {
                int[] raw = r.readF32(numel);
                for (int j = 0; j < numel; j++) arr[j] = Float.intBitsToFloat(raw[j]);
            } else {
                int[] raw = r.readF16(numel);
                for (int j = 0; j < numel; j++) arr[j] = BinIO.f16ToFloat(raw[j]);
            }
            w.put(name, arr);
        }
        int vsz = r.readU32();
        BinIO.expect(vsz, vocab, "vocab count");
        String[] vt = new String[vsz];
        for (int i = 0; i < vsz; i++) vt[i] = r.readString();
        int mc = r.readU32();
        String[][] merges = new String[mc][2];
        for (int i = 0; i < mc; i++) {
            int la = r.readU16();
            int lb = r.readU16();
            merges[i][0] = new String(r.readBytes(la), java.nio.charset.StandardCharsets.UTF_8);
            merges[i][1] = new String(r.readBytes(lb), java.nio.charset.StandardCharsets.UTF_8);
        }
        int[] bm = r.readF32(256);
        int[] bytemap = new int[256];
        for (int i = 0; i < 256; i++) bytemap[i] = bm[i];
        r.close();

        Tokenizer tok = new Tokenizer(vt, merges, bytemap, padId, unkId, bosId, eosId);
        checkDims(w, vocab, L, C, H);
        return new Rwkv(vocab, L, C, H, block, padId, unkId, bosId, eosId, tok, w);
    }

    private static void checkDims(Map<String, float[]> w, int V, int L, int C, int H) {
        float[] wte = w.get("lm_head.weight");
        if (wte == null) throw new IllegalArgumentException("missing lm_head.weight");
        if (wte.length != V * C) throw new IllegalStateException("lm_head.weight bad len " + wte.length);
    }

    public void reset() {
        java.util.Arrays.fill(prevLn1, 0f);
        java.util.Arrays.fill(prevLn2, 0f);
        java.util.Arrays.fill(sd, 0f);
    }

    private float[] layerNorm(float[] x, float[] w) {
        int n = x.length;
        float m = 0;
        for (int i = 0; i < n; i++) m += x[i];
        m /= n;
        float v = 0;
        for (int i = 0; i < n; i++) {
            float d = x[i] - m;
            v += d * d;
        }
        v /= n;
        float inv = (float) (1.0 / Math.sqrt(v + 1e-5f));
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = (x[i] - m) * inv * w[i];
        return out;
    }

    private final float[] tmpA = new float[256 * 4]; // scratch buffers allocated on demand below

    public void forwardOne(int token, float[] logits, float[] xOut) {
        int C = nEmb;
        float[] x = new float[C];
        int te = token * C;
        for (int c = 0; c < C; c++) x[c] = wte[te + c];

        for (int l = 0; l < nLayer; l++) {
            float[] n1 = layerNorm(x, wln1[l]);
            float[] tm = tmix(n1, l);
            x = addTo(x, tm);
            setShift(prevLn1, l, n1);

            float[] n2 = layerNorm(x, wln2[l]);
            float[] cm = cmix(n2, l);
            x = addTo(x, cm);
            setShift(prevLn2, l, n2);
        }

        float[] ln = layerNorm(x, lnF);
        // final head matvec into logits
        for (int i = 0; i < vocab; i++) {
            int off = i * C;
            float s = 0;
            for (int c = 0; c < C; c++) s += wte[off + c] * ln[c];
            logits[i] = s;
        }
        if (xOut != null) System.arraycopy(x, 0, xOut, 0, C);
    }

    public void forwardOne(int token, float[] logits) { forwardOne(token, logits, null); }

    private float[] addTo(float[] x, float[] d) {
        for (int i = 0; i < x.length; i++) x[i] += d[i];
        return x;
    }

    private void setShift(float[] prev, int l, float[] val) {
        int C = nEmb;
        int off = l * C;
        for (int c = 0; c < C; c++) prev[off + c] = val[c];
    }

    /** tmix with use-then-decay per head. */
    private float[] tmix(float[] x, int l) {
        int C = nEmb, H = nHead, N = headSize;
        int lh = l * C;
        float[] shift = new float[C];
        System.arraycopy(prevLn1, lh, shift, 0, C);

        float[] xx = new float[C];
        for (int c = 0; c < C; c++) xx[c] = shift[c] - x[c];

        float[] xk = mix(x, xx, tlma[l], 0);
        float[] xv = mix(x, xx, tlma[l], C);
        float[] xr = mix(x, xx, tlma[l], 2 * C);
        float[] xg = mix(x, xx, tlma[l], 3 * C);

        float[] r = matvec(wtr[l], xr);
        float[] k = matvec(wtk[l], xk);
        float[] v = matvec(wtv[l], xv);
        float[] g = matvec(wtg[l], xg);
        for (int c = 0; c < C; c++) g[c] = silu(g[c]);

        float[] out = new float[C];
        int sdOff = l * H * N * N;
        for (int h = 0; h < H; h++) {
            int hb = sdOff + h * N * N;
            int rb = h * N;
            float rk = 0;
            for (int n = 0; n < N; n++) rk += r[rb + n] * k[rb + n];
            float u = faaa[l * H + h];
            float dec = decay[l * H + h];
            for (int n = 0; n < N; n++) {
                float s = 0;
                for (int m = 0; m < N; m++) s += sd[hb + m * N + n] * r[rb + m];
                out[rb + n] = s + u * rk * v[rb + n];
            }
            // use-then-decay: update state after computing all heads outputs
            for (int m = 0; m < N; m++) {
                float km = k[rb + m], vr = v[rb + m];
                int rowb = hb + m * N;
                for (int n = 0; n < N; n++) {
                    sd[rowb + n] = dec * sd[rowb + n] + km * v[rb + n];
                }
            }
        }

        // group norm over heads, eps = 6.4e-4 = 1e-5*64, then affine ln_x
        float[] lnxW = wlnxW[l], lnxB = wlnxB[l];
        for (int h = 0; h < H; h++) {
            int b = h * N;
            float m = 0;
            for (int n = 0; n < N; n++) m += out[b + n];
            m /= N;
            float vv = 0;
            for (int n = 0; n < N; n++) {
                float d = out[b + n] - m;
                vv += d * d;
            }
            vv /= N;
            float inv = (float) (1.0 / Math.sqrt(vv + 6.4e-4f));
            for (int n = 0; n < N; n++) {
                out[b + n] = (out[b + n] - m) * inv * lnxW[b + n] + lnxB[b + n];
            }
        }
        for (int c = 0; c < C; c++) out[c] *= g[c];
        out = matvec(wto[l], out);
        return out;
    }

    private float[] mix(float[] x, float[] xx, float[] ma, int off) {
        int C = nEmb;
        float[] o = new float[C];
        for (int c = 0; c < C; c++) o[c] = x[c] + xx[c] * ma[off + c];
        return o;
    }

    private float[] cmix(float[] x, int l) {
        int C = nEmb;
        float[] xx = new float[C];
        int lh = l * C;
        for (int c = 0; c < C; c++) xx[c] = prevLn2[lh + c] - x[c];
        float[] xk = mix(x, xx, clma[l], 0);
        float[] xr = mix(x, xx, clma[l], C);
        float[] z = matvec(cmk1[l], xk);
        for (int c = 0; c < z.length; c++) z[c] = (z[c] > 0 ? z[c] : 0) * (z[c] > 0 ? z[c] : 0);
        z = matvec(cmk2[l], z);
        float[] rz = matvec(cmr[l], xr);
        for (int c = 0; c < rz.length; c++) rz[c] = sigmoid(rz[c]);
        for (int c = 0; c < C; c++) rz[c] *= z[c];
        return rz;
    }

    private static float sigmoid(float z) {
        if (z >= 30f) return 1f;
        if (z <= -30f) return 0f;
        return (float) (1.0 / (1.0 + Math.exp(-z)));
    }

    private static float silu(float z) { return z * sigmoid(z); }

    private static float[] matvec(float[] W, float[] x) {
        int in = x.length;
        int out = W.length / in;
        float[] r = new float[out];
        for (int i = 0; i < out; i++) {
            float s = 0;
            int off = i * in;
            for (int j = 0; j < in; j++) s += W[off + j] * x[j];
            r[i] = s;
        }
        return r;
    }
}