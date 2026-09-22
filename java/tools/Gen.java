import com.mystica.core.Generator;
import com.mystica.core.Rwkv;

import java.io.FileInputStream;
import java.io.InputStream;

/** JVM generation harness (mirror of the Android app path) for cross-check.
 *  Usage: Gen <model.bin> <prompt> <seed> <temp> <topk> <maxnew> */
public class Gen {
    public static void main(String[] args) throws Exception {
        String modelFile = args[0];
        String prompt = args[1];
        long seed = Long.parseLong(args[2]);
        float temp = Float.parseFloat(args[3]);
        int topk = Integer.parseInt(args[4]);
        int maxNew = Integer.parseInt(args[5]);

        InputStream is = new FileInputStream(modelFile);
        Rwkv m = Rwkv.load(is);
        Generator g = new Generator(m);
        g.seed(seed);

        int[] ids = m.tok.encode("Вопрос: " + prompt + "\nОтвет:");
        System.out.print("PROMPT_IDS=");
        for (int i : ids) System.out.print(i + " ");
        System.out.println();
        if (ids.length > m.blockSize - 1) {
            int[] cut = new int[m.blockSize - 1];
            System.arraycopy(ids, ids.length - cut.length, cut, 0, cut.length);
            ids = cut;
        }
        m.reset();
        float[] logits = new float[m.vocab];
        for (int id : ids) m.forwardOne(id, logits);

        if (args.length > 6 && args[6].equals("step")) {
            for (int step = 0; step < 40; step++) {
                int best = 0;
                for (int i = 1; i < m.vocab; i++) if (logits[i] > logits[best]) best = i;
                System.out.println("step=" + step + " argmax=" + best +
                        "(='" + m.tok.decodeOne(best) + "') top5=" + topN(logits, m, 5));
                if (best == m.eosId) break;
                m.forwardOne(best, logits);
            }
            return;
        }

        StringBuilder out = new StringBuilder();
        long t0 = System.currentTimeMillis();
        for (int n = 0; n < maxNew; n++) {
            int t = g.nextToken(logits, temp, topk);
            if (t == m.eosId) break;
            out.append(m.tok.decodeOne(t));
            m.forwardOne(t, logits);
        }
        long ms = System.currentTimeMillis() - t0;
        String t = out.toString();
        int qp = t.indexOf("\nВопрос:");
        if (qp >= 0) t = t.substring(0, qp);
        if (t.startsWith("Ответ:")) t = t.substring("Ответ:".length()).trim();
        System.out.println("GEN len=" + t.length() + " ms=" + ms);
        System.out.println("HEAD=" + t.substring(0, Math.min(200, t.length())));
    }

    static String topN(float[] logits, Rwkv m, int n) {
        java.util.PriorityQueue<Integer> q = new java.util.PriorityQueue<>(
                (a, b) -> Float.compare(logits[a], logits[b]));
        for (int i = 0; i < m.vocab; i++) {
            q.add(i);
            if (q.size() > n) q.poll();
        }
        StringBuilder sb = new StringBuilder("[");
        int[] arr = new int[n];
        for (int i = n - 1; i >= 0; i--) arr[i] = q.poll();
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }
}