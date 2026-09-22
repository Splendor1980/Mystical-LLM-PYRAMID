import com.mystica.core.Rwkv;

import java.io.File;
import java.io.FileInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * JVM verification for the Java port.
 *  1) tokenizer.encode vs Python reference (tok_ref.tsv)
 *  2) tokenizer.decode round-trips vs Python idem
 *  3) recurrence logits vs Python reference (ref_ids.txt / ref_logits.bin)
 * Usage: java PortCheck <assetsDir> [chunks]
 */
public class PortCheck {
    static int fails = 0;

    record Case(String in, int[] ids, String dec) {}

    public static void main(String[] args) throws Exception {
        File assets = new File(args[0]);
        Rwkv m = Rwkv.load(new FileInputStream(new File(assets, "model.bin")));

        if (args.length > 1 && args[1].equals("chunks")) {
            for (String s : new String[]{"«ёлочка", "»", "−", "°", "…«ёлочка» — тире", "кавычки «ёлочки» и \"лапки\"",
                    "Погода − 5 °C"}) {
                System.out.println("IN: " + s);
                for (String[] row : m.tok.debugBpe(s)) {
                    System.out.println("   " + String.join(" | ", row)
                            .replace("\n", "\\n"));
                }
                System.out.println("   ids: " + java.util.Arrays.toString(m.tok.encode(s)));
            }
            return;
        }

        List<Case> cases = new ArrayList<>();
        for (String line : Files.readAllLines(new File(assets, "tok_ref.tsv").toPath(), StandardCharsets.UTF_8)) {
            String[] p = line.split("\t", -1);
            String in = new String(Base64.getDecoder().decode(p[0]), StandardCharsets.UTF_8);
            String[] idStr = p[1].trim().isEmpty() ? new String[0] : p[1].trim().split(" ");
            int[] ids = new int[idStr.length];
            for (int i = 0; i < idStr.length; i++) ids[i] = Integer.parseInt(idStr[i]);
            String dec = p.length > 2 && !p[2].isEmpty()
                    ? new String(Base64.getDecoder().decode(p[2]), StandardCharsets.UTF_8) : "";
            cases.add(new Case(in, ids, dec));
        }

        int cp = 0;
        for (Case c : cases) {
            int[] got = m.tok.encode(c.in);
            if (!java.util.Arrays.equals(got, c.ids)) {
                System.out.println("ENC MISMATCH " + c.in.replace("\n", "\\n"));
                System.out.println("  python: " + java.util.Arrays.toString(c.ids));
                System.out.println("  java:   " + java.util.Arrays.toString(got));
                fails++;
            } else cp++;
        }
        System.out.println("encode: " + cp + "/" + cases.size() + " ok");

        int dr = 0;
        for (Case c : cases) {
            if (c.ids.length == 0) continue;
            String back = m.tok.decode(c.ids);
            if (!back.equals(c.dec)) {
                System.out.println("DEC MISMATCH " + c.in.replace("\n", "\\n")
                        + " -> " + back.replace("\n", "\\n") + "  ids=" + java.util.Arrays.toString(c.ids));
                for (int id : c.ids) {
                    String t = m.tok.vocabToken(id);
                    System.out.println("   id " + id + " tok=" + t.replace("\n", "\\n")
                            + " ascii=" + java.util.Arrays.toString(t.codePoints().toArray()));
                }
                fails++;
            } else dr++;
        }
        System.out.println("decode: " + dr + " ok");

        String idLine = Files.readString(new File(assets, "ref_ids.txt").toPath()).trim();
        String[] parts = idLine.split(" ");
        int[] ids = new int[parts.length];
        for (int i = 0; i < parts.length; i++) ids[i] = Integer.parseInt(parts[i]);
        byte[] refBytes = Files.readAllBytes(new File(assets, "ref_logits.bin").toPath());
        float[] ref = new float[refBytes.length / 4];
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(refBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buf.asFloatBuffer().get(ref);

        m.reset();
        float[] logits = new float[m.vocab];
        for (int id : ids) m.forwardOne(id, logits);

        double maxAbs = 0, maxRel = 0;
        int maxAbsAt = -1;
        for (int i = 0; i < m.vocab; i++) {
            double d = Math.abs(logits[i] - ref[i]);
            if (d > maxAbs) { maxAbs = d; maxAbsAt = i; }
            double rel = Math.abs(ref[i]) > 1 ? d / Math.abs(ref[i]) : d;
            if (rel > maxRel) maxRel = rel;
        }
        System.out.println(String.format("recurrence: ids=%d maxAbs=%.3e @%d  maxRel=%.3e",
                ids.length, maxAbs, maxAbsAt, maxRel));
        System.out.println("sample java logits[0..5]: " + java.util.Arrays.toString(java.util.Arrays.copyOfRange(logits, 0, 6)));
        System.out.println("sample py   ref[0..5]:    " + java.util.Arrays.toString(java.util.Arrays.copyOfRange(ref, 0, 6)));

        int k = 10;
        int[] tRef = topK(ref, k), tGot = topK(logits, k);
        System.out.println("top python: " + java.util.Arrays.toString(tRef));
        System.out.println("top java:   " + java.util.Arrays.toString(tGot));
        int common = 0;
        for (int a : tGot) for (int b : tRef) if (a == b) common++;
        System.out.println("top overlap: " + common + "/" + k);

        System.out.println(fails == 0 ? "ALL OK" : ("FAILS: " + fails));
        System.exit(fails == 0 && maxAbs < 0.02 ? 0 : 1);
    }

    static int[] topK(float[] v, int k) {
        int[] idx = new int[k];
        for (int i = 0; i < k; i++) {
            float best = Float.NEGATIVE_INFINITY;
            int bi = -1;
            for (int j = 0; j < v.length; j++) {
                boolean used = false;
                for (int a = 0; a < i; a++) if (idx[a] == j) used = true;
                if (!used && v[j] > best) { best = v[j]; bi = j; }
            }
            idx[i] = bi;
        }
        return idx;
    }
}