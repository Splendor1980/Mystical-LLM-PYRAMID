package com.mystica.core;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** Little-endian binary reader over model.bin. */
public final class BinIO {
    private final DataInputStream in;

    public BinIO(InputStream is) {
        in = new DataInputStream(is);
    }

    public int readU8() throws IOException { return in.readUnsignedByte(); }

    public int readU16() throws IOException {
        int a = in.readUnsignedByte(), b = in.readUnsignedByte();
        return a | (b << 8);
    }

    public int readU32() throws IOException {
        int a = in.readUnsignedByte(), b = in.readUnsignedByte();
        int c = in.readUnsignedByte(), d = in.readUnsignedByte();
        return a | (b << 8) | (c << 16) | (d << 24);
    }

    public long readI64() throws IOException {
        long r = 0;
        for (int i = 0; i < 8; i++) r |= ((long) in.readUnsignedByte()) << (8 * i);
        return r;
    }

    public int[] readF32(int n) throws IOException {
        int[] out = new int[n];
        byte[] buf = new byte[4 * n];
        in.readFully(buf);
        for (int i = 0; i < n; i++) {
            int b0 = buf[4 * i] & 0xff, b1 = buf[4 * i + 1] & 0xff;
            int b2 = buf[4 * i + 2] & 0xff, b3 = buf[4 * i + 3] & 0xff;
            out[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
        return out;
    }

    public int[] readF16(int n) throws IOException {
        int[] out = new int[n];
        byte[] buf = new byte[2 * n];
        in.readFully(buf);
        for (int i = 0; i < n; i++) {
            int b0 = buf[2 * i] & 0xff, b1 = buf[2 * i + 1] & 0xff;
            out[i] = b0 | (b1 << 8);
        }
        return out;
    }

    public String readString() throws IOException {
        int n = readU16();
        byte[] buf = new byte[n];
        in.readFully(buf);
        return new String(buf, StandardCharsets.UTF_8);
    }

    public byte[] readBytes(int n) throws IOException {
        byte[] b = new byte[n];
        in.readFully(b);
        return b;
    }

    public void close() throws IOException { in.close(); }

    public static float f16ToFloat(int h) {
        int s = (h >>> 15) & 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
        float v;
        if (e == 0) {
            v = (m == 0) ? 0f : m * 5.960464477539063e-8f; // m * 2^-24
        } else if (e == 0x1f) {
            v = (m == 0) ? Float.POSITIVE_INFINITY : Float.NaN;
        } else {
            v = (1f + m * 0.0009765625f) * pow2f(e - 15);
        }
        return (s != 0) ? -v : v;
    }

    private static float pow2f(int e) {
        return Float.intBitsToFloat((e + 127) << 23);
    }

    public static void expect(int actual, int want, String what) {
        if (actual != want) throw new IllegalArgumentException(
                String.format("bad %s: got %d, want %d", what, actual, want));
    }

    public static void expectString(String actual, String want, String what) {
        if (!actual.equals(want)) throw new IllegalArgumentException(
                String.format("bad %s: got %s, want %s", what, actual, want));
    }
}