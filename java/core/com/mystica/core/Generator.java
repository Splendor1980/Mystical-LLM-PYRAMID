package com.mystica.core;

import java.util.Random;

/** Temperature + top-k multinomial sampling over logits. */
public final class Generator {
    private final Rwkv rwkv;
    private final Random rng = new Random();

    public Generator(Rwkv rwkv) { this.rwkv = rwkv; }

    public void seed(long s) { rng.setSeed(s); }

    public int nextToken(float[] logits, float temp, int topK) {
        int V = rwkv.vocab;
        double t = temp < 1e-4f ? 0.0 : temp;
        int[] idx = new int[V];
        float[] p = new float[V];
        for (int i = 0; i < V; i++) {
            idx[i] = i;
            p[i] = logits[i];
        }
        if (t > 0) {
            double max = p[0];
            for (int i = 1; i < V; i++) if (p[i] > max) max = p[i];
            for (int i = 0; i < V; i++) {
                p[i] = (float) Math.exp((p[i] - max) / t);
            }
        } else {
            int best = 0;
            for (int i = 1; i < V; i++) if (p[i] > p[best]) best = i;
            return best;
        }
        // partial top-k via selection (quickselect-free, V small)
        if (topK > 0 && topK < V) {
            for (int i = 0; i < topK; i++) {
                int b = i;
                for (int j = i + 1; j < V; j++) if (p[j] > p[b]) b = j;
                float tmp = p[i]; p[i] = p[b]; p[b] = tmp;
                int ti = idx[i]; idx[i] = idx[b]; idx[b] = ti;
            }
            V = topK;
        }
        double sum = 0;
        for (int i = 0; i < V; i++) sum += p[i];
        double r = rng.nextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < V; i++) {
            acc += p[i];
            if (r < acc) return idx[i];
        }
        return idx[V - 1];
    }
}