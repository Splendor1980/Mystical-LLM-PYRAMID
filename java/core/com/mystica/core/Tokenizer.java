package com.mystica.core;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ByteLevel-BPE tokenizer port (tokenizers Rust semantics).
 * Pipeline: NFKC -> GPT2 regex pre-tokenize -> byte-level encode -> BPE merges -> ids.
 */
public final class Tokenizer {
    private final String[] vocab;          // id -> token string
    private final Map<String, Integer> vocabId;
    private final Map<String, Integer> mergeRank; // "a b" -> rank 0..n-1
    private final int[] byteToChar;        // byte -> codepoint (256)
    private final int[] charToByte;        // cp -> byte, only where "encoded" (else -1)
    private final int padId, unkId, bosId, eosId;

    public Tokenizer(String[] vocab, String[][] merges, int[] byteToChar, int padId, int unkId, int bosId, int eosId) {
        this.vocab = vocab;
        this.byteToChar = byteToChar;
        this.padId = padId;
        this.unkId = unkId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.vocabId = new HashMap<>();
        for (int i = 0; i < vocab.length; i++) vocabId.put(vocab[i], i);
        this.mergeRank = new HashMap<>();
        for (int i = 0; i < merges.length; i++) {
            mergeRank.put(merges[i][0] + " " + merges[i][1], i);
        }
        this.charToByte = new int[65536];
        java.util.Arrays.fill(charToByte, -1);
        for (int b = 0; b < 256; b++) {
            int cp = byteToChar[b];
            if (cp != b) charToByte[cp] = b; // only encode-spanned bytes decode back
        }
    }

    public int vocabSize() { return vocab.length; }
    public int unkId() { return unkId; }
    public int padId() { return padId; }

    public String vocabToken(int id) { return vocab[id]; }

    /** raw regex chunks -> byte-encoded strings -> BPE token strings. */
    public String[][] debugBpe(String text) {
        String norm = Normalizer.normalize(text, Normalizer.Form.NFKC);
        List<String> chunks = preTokenize(norm);
        List<String[]> out = new ArrayList<>();
        for (int ci = 0; ci < chunks.size(); ci++) {
            String enc = byteEncode(chunks.get(ci));
            List<String> syms = new ArrayList<>();
            for (int i = 0; i < enc.length(); i++) syms.add(enc.substring(i, i + 1));
            String[] row = new String[3 + syms.size()];
            row[0] = "chunk=" + chunks.get(ci);
            row[1] = "enc=" + enc;
            StringBuilder trace = new StringBuilder("bpe=");
            String[] traceSym = null;
            while (syms.size() >= 2) {
                String best = null;
                int bestRank = Integer.MAX_VALUE;
                for (int i = 0; i < syms.size() - 1; i++) {
                    Integer r = mergeRank.get(syms.get(i) + " " + syms.get(i + 1));
                    if (r != null && r < bestRank) { bestRank = r; best = syms.get(i) + " " + syms.get(i + 1); }
                }
                if (best == null) break;
                String merged = best.substring(0, best.indexOf(' ')) + best.substring(best.indexOf(' ') + 1);
                List<String> ns = new ArrayList<>(syms.size());
                for (int i = 0; i < syms.size(); ) {
                    if (i < syms.size() - 1 && best.equals(syms.get(i) + " " + syms.get(i + 1))) { ns.add(merged); i += 2; }
                    else { ns.add(syms.get(i)); i++; }
                }
                syms = ns;
            }
            traceSym = syms.toArray(new String[0]);
            trace.append(String.join(" | ", traceSym));
            row[2] = trace.toString();
            System.arraycopy(traceSym, 0, row, 3, traceSym.length);
            out.add(row);
        }
        return out.toArray(new String[0][]);
    }

    // ---------------- encode ----------------

    public int[] encode(String text) {
        String norm = Normalizer.normalize(text, Normalizer.Form.NFKC);
        List<String> chunks = preTokenize(norm);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : chunks) {
            String enc = byteEncode(chunk);
            wordIds(enc, ids);
        }
        int[] out = new int[ids.size()];
        for (int i = 0; i < out.length; i++) out[i] = ids.get(i);
        return out;
    }

    /** GPT2 BPE regex, leftmost-first like Rust regex crate. Chunks only (offsets discarded). */
    static List<String> preTokenize(String s) {
        List<String> out = new ArrayList<>();
        int i = 0, n = s.length();
        while (i < n) {
            // alt1: 's|'t|'re|'ve|'m|'ll|'d
            if (i + 1 < n && s.charAt(i) == '\'') {
                char c1 = s.charAt(i + 1);
                if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                    out.add(s.substring(i, i + 2));
                    i += 2;
                    continue;
                }
                if (i + 2 < n && ((c1 == 'r' && s.charAt(i + 2) == 'e')
                        || (c1 == 'v' && s.charAt(i + 2) == 'e')
                        || (c1 == 'l' && s.charAt(i + 2) == 'l'))) {
                    out.add(s.substring(i, i + 3));
                    i += 3;
                    continue;
                }
            }
            int start = i;
            boolean space = false;
            if (s.charAt(i) == ' ') { space = true; i++; }
            if (i < n && isLetter(codePointAt(s, i))) {
                while (i < n && isLetter(codePointAt(s, i))) i = nextCp(s, i);
                out.add(s.substring(start, i));
                continue;
            }
            i = start;
            if (s.charAt(i) == ' ') { i++; }
            if (i < n && isNumber(codePointAt(s, i))) {
                while (i < n && isNumber(codePointAt(s, i))) i = nextCp(s, i);
                out.add(s.substring(start, i));
                continue;
            }
            i = start;
            if (s.charAt(i) == ' ') { i++; }
            if (i < n && !isWs(codePointAt(s, i)) && !isLetter(codePointAt(s, i)) && !isNumber(codePointAt(s, i))) {
                while (i < n) {
                    int cp = codePointAt(s, i);
                    if (isWs(cp) || isLetter(cp) || isNumber(cp)) break;
                    i = nextCp(s, i);
                }
                out.add(s.substring(start, i));
                continue;
            }
            i = start;
            // ' \s+(?!\S) ' then plain '\s+': leftmost-first + backtrack like Rust regex.
            // A whitespace run directly followed by non-space text leaves ONE space aside
            // so the following token can still use its optional leading space.
            int wj = i;
            while (wj < n && isWs(codePointAt(s, wj))) wj = nextCp(s, wj);
            int wend = wj;
            if (wj == n) {
                out.add(s.substring(start, wj));
                i = wj;
            } else {
                // run [i, wj) followed by non-space: greedy \s+ backtracks until lookahead at
                // a following whitespace succeeds; first such position is wj-1 if run length>=2.
                if (wj - 1 > i) {
                    out.add(s.substring(start, wj - 1));
                    i = wj - 1;
                } else {
                    out.add(s.substring(start, wj));
                    i = wj;
                }
            }
        }
        return out;
    }

    private static int nextCp(String s, int i) {
        return i + Character.charCount(codePointAt(s, i));
    }

    private static int codePointAt(String s, int i) { return s.codePointAt(i); }

    static boolean isLetter(int cp) { return Character.isLetter(cp); }

    static boolean isNumber(int cp) {
        int t = Character.getType(cp);
        return t == Character.DECIMAL_DIGIT_NUMBER || t == Character.LETTER_NUMBER || t == Character.OTHER_NUMBER;
    }

    static boolean isWs(int cp) {
        return cp == 0x20 || cp == 0x9 || cp == 0xA || cp == 0xB || cp == 0xC || cp == 0xD
                || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A)
                || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
    }

    /** byte-level encode: utf-8 bytes of chunk -> mapped chars via byteToChar. */
    private String byteEncode(String chunk) {
        byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.appendCodePoint(byteToChar[b & 0xff]);
        }
        return sb.toString();
    }

    private void wordIds(String word, List<Integer> out) {
        List<String> syms = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            syms.add(word.substring(i, i + 1));
        }
        while (syms.size() >= 2) {
            String best = null;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < syms.size() - 1; i++) {
                Integer r = mergeRank.get(syms.get(i) + " " + syms.get(i + 1));
                if (r != null && r < bestRank) {
                    bestRank = r;
                    best = syms.get(i) + " " + syms.get(i + 1);
                }
            }
            if (best == null) break;
            String merged = best.substring(0, best.indexOf(' ')) + best.substring(best.indexOf(' ') + 1);
            List<String> ns = new ArrayList<>(syms.size());
            for (int i = 0; i < syms.size(); ) {
                if (i < syms.size() - 1 && best.equals(syms.get(i) + " " + syms.get(i + 1))) {
                    ns.add(merged);
                    i += 2;
                } else {
                    ns.add(syms.get(i));
                    i++;
                }
            }
            syms = ns;
        }
        for (String s : syms) {
            Integer id = vocabId.get(s);
            out.add(id != null ? id : unkId);
        }
    }

    // ---------------- decode ----------------

    public String decode(int[] ids) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int id : ids) {
            if (id == padId || id == unkId || id == bosId || id == eosId) continue;
            String tok = vocab[id];
            for (int i = 0; i < tok.length(); ) {
                int cp = tok.codePointAt(i);
                int b = (cp < charToByte.length) ? charToByte[cp] : -1;
                bytes.write(b >= 0 ? b : (cp & 0xff));
                i += Character.charCount(cp);
            }
        }
        return lossyUtf8(bytes.toByteArray());
    }

    /** decode one id at a time to inspect tokens (e.g. stop conditions). */
    public String decodeOne(int id) {
        return decode(new int[]{id});
    }

    private static String lossyUtf8(byte[] data) {
        try {
            return new String(data, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return "\ufffd";
        }
    }
}