package com.mystica.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.mystica.core.Generator;
import com.mystica.core.Rwkv;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/** Офлайн-оракул «Мистика»: юзер пишет, модель отвечает без сети. */
public class MainActivity extends Activity {

    private static final String PREFS = "mystica";
    private static final String KEY_TEMP = "temp";
    private static final String KEY_TOPK = "topk";
    private static final String KEY_MAXNEW = "maxnew";
    private static final String KEY_SEEN = "seen";

    static final String ACTION_TEST = "com.mystica.app.TEST";

    private LinearLayout chat;
    private ScrollView svChat;
    private EditText input;
    private LinearLayout typingRow;
    private TextView typingText;
    private ImageView ivOko;

    private Rwkv model;
    private Generator generator;
    private Handler bg, ui = new Handler(Looper.getMainLooper());
    private boolean ready;
    private boolean generating;
    private String lastPrompt;

    private float temp = 0.7f;
    private int topk = 40;
    private int maxNew = 200;

    private final List<String> chips = new ArrayList<>();
    {
        chips.add("Расскажи о месте силы");
        chips.add("Гадание на рунах");
        chips.add("Тайна старой церкви");
        chips.add("Сон про чёрного кота");
        chips.add("Что ждёт меня в пути");
        chips.add("Легенда о северном лесе");
        chips.add("Значение свечи во сне");
        chips.add("Как услышать интуицию");
    }

    private final StringBuilder pendingBot = new StringBuilder();
    private long genStart;
    private int lastTok = -1;
    private int loopStreak;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        bindViews();

        SharedPreferences p = getSharedPreferences(PREFS, MODE_PRIVATE);
        temp = p.getFloat(KEY_TEMP, 0.7f);
        topk = p.getInt(KEY_TOPK, 40);
        maxNew = p.getInt(KEY_MAXNEW, 200);

        HandlerThread th = new HandlerThread("mystica-bg");
        th.start();
        bg = new Handler(th.getLooper());

        touchTyping(false);
        setOkoThinking(false);
        input.setEnabled(false);
        findViewById(R.id.btnSend).setEnabled(false);

        bg.post(new Runnable() {
            @Override public void run() {
                try {
                    loadModel();
                    ui.post(new Runnable() {
                        @Override public void run() {
                            ready = true;
                            input.setEnabled(true);
                            findViewById(R.id.btnSend).setEnabled(true);
                            setOkoThinking(false);
                            if (!p.getBoolean(KEY_SEEN, false)) {
                                p.edit().putBoolean(KEY_SEEN, true).apply();
                                showOnboarding();
                            }
                        }
                    });
                } catch (final Throwable t) {
                    ui.post(new Runnable() {
                        @Override public void run() {
                            typingText.setText("Не удалось открыть книгу: " + t);
                            toast("Ошибка загрузки модели");
                        }
                    });
                }
            }
        });
    }

    private void bindViews() {
        chat = findViewById(R.id.llChat);
        svChat = findViewById(R.id.svChat);
        input = findViewById(R.id.edInput);
        typingRow = findViewById(R.id.llTyping);
        typingText = findViewById(R.id.tvTyping);
        ivOko = findViewById(R.id.ivOko);

        ImageButton send = findViewById(R.id.btnSend);
        send.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { submit(); }
        });
        input.setOnEditorActionListener((tv, actionId, ev) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND || actionId == EditorInfo.IME_ACTION_DONE) {
                submit();
                return true;
            }
            return false;
        });
        findViewById(R.id.btnSettings).setOnClickListener(v -> showSettings());
        findViewById(R.id.btnReload).setOnClickListener(v -> retry());

        addChips();
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleTestIntent(intent);
    }

    private void handleTestIntent(Intent i) {
        if (i == null || !ACTION_TEST.equals(i.getAction())) return;
        if (!ready || generating) return;
        String b64 = i.getStringExtra("prompt_b64");
        if (b64 == null) return;
        String prompt;
        try {
            prompt = new String(android.util.Base64.decode(b64, android.util.Base64.NO_WRAP),
                    "UTF-8");
        } catch (Throwable t) {
            return;
        }
        int mn = i.getIntExtra("maxnew", 0);
        if (mn > 0) maxNew = Math.min(mn, 512);
        try {
            String s = i.getStringExtra("seed");
            if (s != null) generator.seed(Long.parseLong(s));
            String t = i.getStringExtra("temp");
            if (t != null) temp = Float.parseFloat(t);
            String k = i.getStringExtra("topk");
            if (k != null) topk = Integer.parseInt(k);
        } catch (Throwable ignored) { }
        lastPrompt = prompt;
        addUser(prompt);
        generate(prompt);
        android.util.Log.i("MysticaTest", "TEST_PROMPT len=" + prompt.length());
    }

    private void addChips() {
        LinearLayout row = findViewById(R.id.llChips);
        for (final String c : chips) {
            TextView b = new TextView(this);
            b.setText(c);
            b.setTextColor(Color.rgb(245, 237, 220));
            b.setTextSize(13f);
            b.setGravity(Gravity.CENTER);
            b.setPadding(dp(14), dp(8), dp(14), dp(8));
            b.setBackgroundResource(R.drawable.bg_input);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            lp.setMargins(dp(6), 0, dp(6), 0);
            row.addView(b, lp);
            b.setOnClickListener(v -> {
                input.setText(c);
                input.setSelection(c.length());
                submit();
            });
        }
    }

    private void submit() {
        if (!ready || generating) return;
        String q = input.getText().toString().trim();
        if (q.isEmpty()) return;
        input.setText("");
        lastPrompt = q;
        addUser(q);
        generate(q);
    }

    private void retry() {
        if (!ready || generating) return;
        if (lastPrompt == null) {
            toast("Сначала задайте вопрос");
            return;
        }
        generate(lastPrompt);
    }

    private void generate(final String query) {
        generating = true;
        genStart = android.os.SystemClock.elapsedRealtime();
        findViewById(R.id.btnSend).setEnabled(false);
        findViewById(R.id.btnReload).setEnabled(false);
        configChips(false);
        setOkoThinking(true);
        typingRow.setVisibility(View.VISIBLE);
        typingText.setText(R.string.status_think);

        final TextView bot = addBot("");
        bot.setMovementMethod(ScrollingMovementMethod.getInstance());
        pendingBot.setLength(0);

        bg.post(new Runnable() {
            @Override public void run() {
                try {
                    StringBuilder out = new StringBuilder();
                    String templated = "Вопрос: " + query + "\nОтвет:";
                    int[] ids = model.tok.encode(templated);
                    if (ids.length > model.blockSize - 1) {
                        int[] cut = new int[model.blockSize - 1];
                        System.arraycopy(ids, ids.length - cut.length, cut, 0, cut.length);
                        ids = cut;
                    }
                    model.reset();
                    float[] logits = new float[model.vocab];
                    for (int id : ids) model.forwardOne(id, logits);
                    lastTok = -1;
                    loopStreak = 0;

                    for (int n = 0; n < maxNew; n++) {
                        int t = generator.nextToken(logits, temp, topk);
                        if (t == model.eosId) break;
                        loopStreak = (t == lastTok) ? loopStreak + 1 : 0;
                        lastTok = t;
                        if (loopStreak >= 12) break;
                        out.append(model.tok.decodeOne(t));
                        model.forwardOne(t, logits);
                        if (n % 8 == 0 || n == maxNew - 1) {
                            final String chunk = out.toString();
                            ui.post(() -> {
                                pendingBot.setLength(0);
                                pendingBot.append(chunk);
                                bot.setText(chunk + " …");
                                scrollChat();
                            });
                        }
                    }
                    String finalText = out.toString();
                    int qp = finalText.indexOf("\nВопрос:");
                    if (qp >= 0) finalText = finalText.substring(0, qp);
                    if (finalText.startsWith("Ответ:")) {
                        finalText = finalText.substring("Ответ:".length()).trim();
                    }
                    final long ms = android.os.SystemClock.elapsedRealtime();
                    final String finalTextFinal = finalText;
                    ui.post(() -> {
                        bot.setText(finalTextFinal);
                        pendingBot.setLength(0);
                        pendingBot.append(finalTextFinal);
                        scrollChat();
                        doneGenerating();
                        android.util.Log.i("MysticaTest", "TEST_RESP len=" + finalTextFinal.length()
                                + " head=" + (finalTextFinal.length() > 80 ? finalTextFinal.substring(0, 80) : finalTextFinal)
                                + " ms=" + (ms - genStart));
                    });
                } catch (final Throwable t) {
                    ui.post(() -> {
                        bot.setText("Око затуманилось: " + t);
                        doneGenerating();
                    });
                }
            }
        });
    }

    private void doneGenerating() {
        if (!isFinishing()) setOkoThinking(false);
        typingRow.setVisibility(View.GONE);
        generating = false;
        findViewById(R.id.btnSend).setEnabled(true);
        findViewById(R.id.btnReload).setEnabled(true);
        configChips(true);
    }

    private TextView addUser(String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextSize(17f);
        t.setLineSpacing(0f, 1.1f);
        t.setTextColor(Color.rgb(245, 237, 220));
        t.setBackgroundResource(R.drawable.bg_bubble_user);
        t.setPadding(dp(14), dp(12), dp(14), dp(12));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.setMargins(dp(48), dp(4), dp(4), dp(8));
        lp.gravity = Gravity.END;
        chat.addView(t, lp);
        scrollChat();
        return t;
    }

    private TextView addBot(String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextIsSelectable(true);
        t.setTextSize(17f);
        t.setLineSpacing(0f, 1.15f);
        t.setTextColor(Color.rgb(245, 237, 220));
        t.setBackgroundResource(R.drawable.bg_bubble_bot);
        t.setPadding(dp(14), dp(12), dp(14), dp(12));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        lp.setMargins(dp(4), dp(4), dp(40), dp(8));
        chat.addView(t, lp);
        scrollChat();
        return t;
    }

    private void scrollChat() {
        svChat.post(() -> svChat.fullScroll(View.FOCUS_DOWN));
    }

    private void setOkoThinking(boolean thinking) {
        ivOko.clearAnimation();
        if (thinking) {
            android.animation.ObjectAnimator a = android.animation.ObjectAnimator
                    .ofFloat(ivOko, "alpha", 0.35f, 1f);
            a.setDuration(700);
            a.setRepeatCount(android.animation.ValueAnimator.INFINITE);
            a.setRepeatMode(android.animation.ValueAnimator.REVERSE);
            a.start();
        } else {
            ivOko.setAlpha(1f);
        }
    }

    private void configChips(boolean on) {
        LinearLayout row = findViewById(R.id.llChips);
        for (int i = 0; i < row.getChildCount(); i++) row.getChildAt(i).setEnabled(on);
    }

    private void touchTyping(boolean on) {
        typingRow.setVisibility(on ? View.VISIBLE : View.GONE);
    }

    // ---------- settings ----------

    private void showSettings() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(20), dp(16), dp(20), dp(8));

        SeekBar tempBar = slider(box, "Мера откровения (жар ответа)", 20, 120, Math.round(temp * 100f));
        SeekBar topkBar = slider(box, "Круг выбора (top к)", 1, 100, (int) topk);
        SeekBar maxBar = slider(box, "Длина ответа (слов)", 16, 512, (int) maxNew);

        AlertDialog d = new AlertDialog.Builder(this)
                .setTitle(R.string.dlg_title)
                .setView(box)
                .setNegativeButton(R.string.dlg_cancel, null)
                .setPositiveButton(R.string.dlg_save, (dlg, which) -> {
                    temp = tempBar.getProgress() / 100f;
                    topk = topkBar.getProgress();
                    maxNew = maxBar.getProgress();
                    getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                            .putFloat(KEY_TEMP, temp)
                            .putInt(KEY_TOPK, topk)
                            .putInt(KEY_MAXNEW, maxNew)
                            .apply();
                })
                .create();
        d.show();
    }

    private SeekBar slider(LinearLayout box, final String label, int min, int max, int val) {
        TextView tv = new TextView(this);
        tv.setText(label);
        tv.setTextColor(Color.rgb(245, 237, 220));
        tv.setTextSize(14f);
        tv.setPadding(0, dp(8), 0, dp(4));
        SeekBar bar = new SeekBar(this);
        bar.setMax(max - min);
        bar.setProgress(val - min);
        box.addView(tv);
        box.addView(bar);
        bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar sb, int p, boolean f) {
                int v = p + min;
                if (label.startsWith("Мера")) {
                    tv.setText(label + ": " + String.format("%.1f", v / 100f));
                } else if (label.startsWith("Круг")) {
                    tv.setText(label + ": " + v);
                } else {
                    tv.setText(label + ": " + v);
                }
            }
            @Override public void onStartTrackingTouch(SeekBar sb) { }
            @Override public void onStopTrackingTouch(SeekBar sb) { }
        });
        return bar;
    }

    private void showOnboarding() {
        new AlertDialog.Builder(this)
                .setTitle(R.string.onboard_title)
                .setMessage(R.string.onboard_text)
                .setPositiveButton(R.string.onboard_ok, null)
                .show();
    }

    private void loadModel() throws IOException {
        InputStream is = getAssets().open("model.bin");
        model = Rwkv.load(is);
        generator = new Generator(model);
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density + 0.5f);
    }
}