# model/

| Файл | Размер | Формат | Назначение | SHA-256 |
|---|---|---|---|---|
| `model.bin` | 22.2 МБ | fp16, формат `RWKV5` | инференс (Java/Python) | `F7273D1DE380F76D938517011C18089BF318AA56D6832C21B686169DA6666FD2` |
| `ckpt_best.pt` | ~126.5 МБ | fp32, PyTorch state_dict | продолжение обучения / эксперименты | `D8D8886FEDDFD60D4FB013A2A4DD25F215B9D9456759409F9E56DD64997B075A` |
| `ckpt_last.pt` | ~126.5 МБ | fp32, PyTorch state_dict | то же (последний чекпойнт) | `419BBF144649D8F7EE728B0FE6A2AB044ADA77BF527D71231782EBC2E143EA67` |

## Формат model.bin

Бинарный, `RWKV5` + версия + размеры + тензоры (fp16; `time_decay`/`time_faaaa` —
fp32) + токенизатор (BPE + merges + bytemap). Полный спек расказан в
`../training/export_app_model.py`. Файл переиспользуем из Java-порта напрямую.

## Генерация (настройки)

`temperature 0.6–0.8`, `top_k 20–40`; стоп — токен `<eos>` (id 3).
Промпт в формате `Вопрос: {текст}\nОтвет:` (без `?` в тексте вопроса).