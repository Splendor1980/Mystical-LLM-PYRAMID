"""Regenerate bytemap.json with the verified OpenAI bytes_to_unicode mapping.

Verified against the tokenizer's vocab/merges: the ONLY non-self char outside
this map is U+0143 ('Ń' = byte 0xFF), which the map includes.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_JSON = os.path.join(ROOT, 'Синтетика от Соннет 5', 'tokenizer.json')
OUT = os.path.join(ROOT, 'kaggle', 'bytemap.json')


def bytes_to_unicode():
    m = {}
    for b in range(0x100):
        if b <= 0x20:
            m[b] = 0x100 + b
        elif b <= 0x7E:
            m[b] = b
        elif b <= 0xA0:
            m[b] = 0x100 + (b - 0x7F + 0x21)
        elif b == 0xAD:
            m[b] = 0x143
        else:
            m[b] = b
    return m


bm = bytes_to_unicode()
json.dump({str(b): c for b, c in sorted(bm.items())}, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# validation against tokenizer vocab/merges
tok = json.load(open(TOK_JSON, encoding='utf-8'))
v = set(tok['model']['vocab'].keys())
mflat = set()
for a, b in tok['model']['merges']:
    mflat.add(a)
    mflat.add(b)
pred = {chr(c) for c in bm.values()}
present = v | mflat
extra = [hex(ord(k)) for k in present if len(k) == 1 and ord(k) >= 0x101 and k not in pred]
print('predicted chars:', len(bm))
print('extra non-self chars in vocab/merges outside map:', extra)
for b in (0, 0x0a, 0x20, 0x7f, 0x80, 0x9f, 0xa0, 0xab, 0xc0, 0xc2, 0xff):
    print(hex(b), '->', hex(bm[b]), repr(chr(bm[b])))