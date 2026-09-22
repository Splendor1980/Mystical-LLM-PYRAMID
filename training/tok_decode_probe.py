"""Probe: how does Python decode every vocab token? Compare with pure byte-decode hypothesis."""
import json, os

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_JSON = os.path.join(ROOT, 'Синтетика от Соннет 5', 'tokenizer.json')
BYPEMAP = os.path.join(ROOT, 'kaggle', 'bytemap.json')

tok = Tokenizer.from_file(TOK_JSON)
bytemap = {int(k): v for k, v in json.load(open(BYPEMAP, encoding='utf-8')).items()}
# decoder: only entries where the mapped char actually differs from the byte itself
char2byte = {codep: byte for byte, codep in bytemap.items() if codep != byte}

special_ids = {0, 1, 2, 3}
mismatch = []
for i in range(tok.get_vocab_size()):
    s = tok.decode([i])
    if i in special_ids:
        expect = ''
    else:
        tok_str = [k for k, v in tok.get_vocab().items() if v == i][0]
        out = []
        for c in tok_str:
            cp = ord(c)
            if cp in char2byte:
                out.append(chr(char2byte[cp]))
            else:
                out.append(c)
        raw = ''.join(out)
        expect = raw.encode('latin-1', 'replace').decode('utf-8', 'replace')
    if s != expect:
        mismatch.append((i, ascii(s), ascii(expect)))

print('mismatches:', len(mismatch))
for m in mismatch[:20]:
    print(m)