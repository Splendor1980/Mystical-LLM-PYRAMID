"""Reference greedy/sampling generation using the SAME fp16-rounded weights as Java.
Confirms whether the trained model itself degenerates (mush/loops) or the port is to blame."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_app_model import load_state, Rwkv51
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, 'kaggle', 'outputs_v1', 'ckpt', 'ckpt_last.pt')
TOK_JSON = os.path.join(ROOT, 'Синтетика от Соннет 5', 'tokenizer.json')

ck, fixed = load_state(CKPT, None)
args = ck['model_args']

w = {}
for k, v in fixed.items():
    if k.endswith('time_decay') or k.endswith('time_faaaa'):
        w[k] = v
    else:
        w[k] = np.float64(v.astype(np.float16))

tok = Tokenizer.from_file(TOK_JSON)
eos = 3

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def gen(prompt, temp, topk, maxnew):
    ids = tok.encode(prompt).ids
    print('PROMPT_IDS=', ids[:80])
    m = Rwkv51(w, args)
    m.reset()
    for i in ids:
        lg = m.forward_one(i)
    out = []
    for step in range(maxnew):
        t = int(np.argmax(lg))
        top5 = np.argsort(lg)[::-1][:5].tolist()
        print('step=%d argmax=%d(=%r) top5=%s' % (step, t, tok.decode([t]), top5))
        if t == eos:
            break
        out.append(tok.decode([t]))
        lg = m.forward_one(t)
    print('OUT=', repr(''.join(out))[:200])

gen('Око леса смотрит на тебя', 0, 1, 16)