"""Dump tokenizer reference encodes for the Java port check (tok_ref.json)."""
import json
import os

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_JSON = os.path.join(ROOT, 'Синтетика от Соннет 5', 'tokenizer.json')
OUT = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'tok_ref.json')

tok = Tokenizer.from_file(TOK_JSON)

battery = [
    '',
    'Привет',
    'Привет, как дела?\n',
    'мир!',
    '  123',
    'Око не молчит\nновая строка',
    'ТЕСТ: 42,5% (готово)',
    '…«ёлочка» — тире',
    'Мистика — это наука о невидимом.',
    'ё Ё й Й э Э ъ Ь',
    'Привет  мир',
    'a b c d e',
    '.,!?:;()[]{}',
    '0 1 2 3 4 5 6 7 8 9',
    '½ ¾ Ⅳ Ⅷ',
    'ﬁ ﬂ ﬀ ﬃ',
    'ＡＢＣ　１２３',
    'ｶﾀｶﾅ ｽﾍﾟｰｽ',
    'Zepellin pumping up: 10.000 km²',
    'Привет\tмир\nдела',
    '😀 emoji test 🎃',
    'Погода − 5 °C',
    'tmp tmp tmp tmp',
    'что-то это"скифское"',
    'Вопрос: Расскажи о луне\nОтвет:',
    'число 1234567890 и ещё 0.0045',
    'кавычки «ёлочки» и "лапки"',
    'The quick brown fox jumps over the lazy dog.',
    'fullwidth：コロン',
    'desertification',
]

refs = []
for s in battery:
    try:
        ids = tok.encode(s).ids
    except Exception as e:
        ids = ['ERR:' + str(e)]
    dec = tok.decode(ids)
    refs.append({'in': s, 'ids': ids, 'dec': dec})

json.dump(refs, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
with open(OUT.replace('.json', '.tsv'), 'w', encoding='utf-8') as f:
    for r in refs:
        enc = r['in'].encode('utf-8')
        import base64
        f.write(base64.b64encode(enc).decode('ascii') + '\t' + ' '.join(map(str, r['ids'])) + '\n')
print('wrote', OUT, len(refs), 'cases')
for r in refs[:5]:
    print(r)