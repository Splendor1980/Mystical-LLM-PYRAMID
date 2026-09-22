# java/ — порт инференса без Python

Зависимый от одного файла Java-порт токенизатора и рекуррентного инференса
RWKV-5.2 11M. Работает на чистом JVM (без Android SDK).

```
core/com/mystica/core/
  BinIO.java      — чтение model.bin (fp16/fp32, тензоры, токенизатор)
  Tokenizer.java  — BPE + NFKC + ByteLevel + bytemap, encode/decode
  Rwkv.java       — рекуррентный forward (O(1)-state, fp16-веса)
  Generator.java  — сэмплинг (temperature + top-k, seed)
tools/
  PortCheck.java  — верификация порта против Python-эталонов
  Gen.java        — CLI-генерация
```

## Сборка и запуск (PowerShell)

```powershell
$core = Get-ChildItem -Recurse java\core -Filter *.java | % FullName
javac -encoding UTF-8 -d build\classes $core java\tools\Gen.java
java -Xmx512m -Dfile.encoding=UTF-8 -cp build\classes Gen model\model.bin "Какой камень лучше всего подходит для защиты дома" 11 0.7 40 110
```

## Верификация (PortCheck, против эталонов Python)

| Проверка | Результат |
|---|---|
| encode (30 кейсов) | 30/30 |
| decode (3000 id) | 0 расхождений |
| рекуррентность (14 токенов, maxAbs) | 6.2e-06 |
| top-10 логитов | 10/10 |
| e2e детерминизм (ПК ↔ Android) | идентичный текст до FMA-хвоста |

Реализация `forwardOne(token, logits)` — пошагово равна Python (fp16-округлённые
веса). `status`: никаких внешних библиотек.