# android/ — приложение «Мистика» (Gradle-free)

Тёмный «мистический» чат: вопрос-пузырь слева, ответ справа, чипсы-подсказки тем.
Веса грузятся из assets; интернет и разрешения не нужны.

```
src/main/AndroidManifest.xml        # package com.mystica.app, minSdk 26, targetSdk 34
src/main/java/.../MainActivity.java # UI + генерация (порт из ../java)
src/main/res/                       # тема, layout, иконки (без adaptive-маски)
build-apk.ps1                       # сборка подписанного APK без Gradle
```

## Сборка

```powershell
# 1) положить model.bin в src/main/assets/ (берётся из ../model/model.bin)
# 2) собрать:
powershell -ExecutionPolicy Bypass -File build-apk.ps1
# готово: build\android\Mystica-debug.apk
```

Релизная подпись:

```powershell
powershell -ExecutionPolicy Bypass -File build-apk.ps1 `
  -Keystore "keys/mystica-release.keystore" -StorePass "<пароль>" -AliasName mystica
```

Сборщик стадирует источники в ASCII-временную папку (aapt2 не любит кириллические
пути) и копирует готовый APK обратно в `build\android`.

## Как это работает

- Формат промпта строго как в обучении: `Вопрос: {текст}\nОтвет:`.
- Ответ обрезается по метке следующего `Вопрос:`; гуард от зацикливания
  (12 одинаковых токенов).
- Автотест на устройстве: `am start ... -a com.mystica.app.TEST --es prompt_b64 <b64>`.
- Никаких permissions — данные никуда не уходят.