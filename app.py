#!/usr/bin/env python3
"""
WhatsApp Sohbet Analiz Pipeline — app.py
_chat.txt → PNG grafikler → PDF rapor

Kişi adları dosyadan otomatik çekilir; elle giriş gerekmez.
Kullanım:  python app.py [_chat.txt] [çıktı_klasörü]
"""

import re
import sys
import statistics
from datetime import datetime
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ── Sabitler ─────────────────────────────────────────────────────────────────
CHAT_FILE      = "_chat.txt"
OUTPUT_DIR     = Path("chat_analysis")
CONV_GAP_HOURS = 4      # Bu süreden uzun sessizlik = yeni konuşma
LATE_REPLY_MIN = 60     # Geç yanıt eşiği (dakika)

COLORS = ["#4A90D9", "#E85D75", "#4CAF50", "#FF9800"]  # 4'e kadar kişi desteği
DARK   = "#1a1a2e"
LIGHT  = "#f7f9fc"
ACCENT = "#4A90D9"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": LIGHT,
    "figure.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ── Regex ─────────────────────────────────────────────────────────────────────
MSG_RE   = re.compile(
    r"^\u200e?\[(\d{1,2}\.\d{1,2}\.\d{4}),\s*(\d{2}:\d{2}:\d{2})\]\s(.+?):\s(.*)"
)
MEDIA_KW = ("omitted", "<This message was edited>", "end-to-end encrypted")
SKIP_RE  = re.compile(
    r"(Messages and calls are end-to-end encrypted"
    r"|image omitted|video omitted|audio omitted"
    r"|sticker omitted|GIF omitted|Contact card omitted|document omitted)",
    re.IGNORECASE,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. PARSE
# ════════════════════════════════════════════════════════════════════════════
def parse_chat(filepath: str) -> pd.DataFrame:
    """_chat.txt → DataFrame; kişi adları dosyadan otomatik çekilir."""
    rows, current = [], None

    with open(filepath, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").replace("\u200e", "").replace("\u200f", "")
            m = MSG_RE.match(line)
            if m:
                if current:
                    rows.append(current)
                date_s, time_s, sender, text = m.groups()
                # Sistem mesajlarını atla
                if SKIP_RE.search(text):
                    current = None
                    continue
                dt = datetime.strptime(f"{date_s} {time_s}", "%d.%m.%Y %H:%M:%S")
                is_media = any(kw in text for kw in MEDIA_KW)
                current = {
                    "dt": dt, "sender": sender.strip(),
                    "text": text, "is_media": is_media,
                }
            elif current and line.strip():
                current["text"] += " " + line.strip()

    if current:
        rows.append(current)

    df = pd.DataFrame(rows)
    df = df[~df["text"].str.contains("end-to-end encrypted", na=False)].copy()
    df["word_count"] = df.apply(
        lambda r: len(r["text"].split()) if not r["is_media"] else 0, axis=1
    )
    df["char_count"] = df.apply(
        lambda r: len(r["text"]) if not r["is_media"] else 0, axis=1
    )
    df["hour"]    = df["dt"].dt.hour
    df["weekday"] = df["dt"].dt.weekday
    df["date"]    = df["dt"].dt.date
    df["month"]   = df["dt"].dt.to_period("M")
    return df.reset_index(drop=True)


def detect_senders(df: pd.DataFrame) -> list[str]:
    """Mesaj sayısına göre sıralı kişi listesi döner (en çok yazan önce)."""
    counts = df["sender"].value_counts()
    return counts.index.tolist()


# ════════════════════════════════════════════════════════════════════════════
# 2. ANALİZ — istatistik hesapları
# ════════════════════════════════════════════════════════════════════════════
def compute_response_times(df: pd.DataFrame) -> pd.DataFrame:
    rt_rows = []
    for i in range(1, len(df)):
        prev, curr = df.iloc[i - 1], df.iloc[i]
        if prev["sender"] == curr["sender"]:
            continue
        gap = (curr["dt"] - prev["dt"]).total_seconds()
        if 0 < gap <= CONV_GAP_HOURS * 3600:
            rt_rows.append({
                "responder": curr["sender"],
                "asked_by":  prev["sender"],
                "gap_sec":   gap,
                "gap_min":   gap / 60,
                "dt":        curr["dt"],
                "is_late":   gap > LATE_REPLY_MIN * 60,
            })
    return pd.DataFrame(rt_rows)


def compute_starters(df: pd.DataFrame) -> Counter:
    starters = [df.iloc[0]["sender"]]
    for i in range(1, len(df)):
        if (df.iloc[i]["dt"] - df.iloc[i - 1]["dt"]).total_seconds() > CONV_GAP_HOURS * 3600:
            starters.append(df.iloc[i]["sender"])
    return Counter(starters)


def print_summary(df: pd.DataFrame, rt_df: pd.DataFrame,
                  starters: Counter, senders: list[str]) -> None:
    total_days = (df["dt"].max() - df["dt"].min()).days + 1
    print("\n" + "=" * 62)
    print("  SAYISAL ÖZET")
    print("=" * 62)
    print(f"  Kişiler  : {', '.join(senders)}")
    print(f"  Dönem    : {df['dt'].min():%d.%m.%Y} → {df['dt'].max():%d.%m.%Y}  ({total_days} gün)")
    print(f"  Toplam   : {len(df)} mesaj  |  {len(df)/total_days:.1f} mesaj/gün\n")
    for s in senders:
        sub    = df[df["sender"] == s]
        rt_sub = rt_df[rt_df["responder"] == s]["gap_min"] if not rt_df.empty else pd.Series([], dtype=float)
        txt    = sub[~sub["is_media"]]
        print(f"  {s}")
        print(f"    Mesaj        : {len(sub)}  ({len(sub)/len(df)*100:.1f}%)")
        print(f"    Medya        : {sub['is_media'].sum()}")
        if len(rt_sub):
            print(f"    Ort yanıt    : {rt_sub.mean():.0f}dk  |  Med: {rt_sub.median():.0f}dk  |  Maks: {rt_sub.max():.0f}dk")
            print(f"    Geç yanıt    : {(rt_sub > LATE_REPLY_MIN).sum()} kez")
        if len(txt):
            print(f"    Ort kelime   : {txt['word_count'].mean():.1f}")
        print(f"    Konuşma başlattı: {starters[s]} kez")
    print()


# ════════════════════════════════════════════════════════════════════════════
# 3. GRAFİKLER
# ════════════════════════════════════════════════════════════════════════════
def _colors(senders: list[str]) -> list[str]:
    return COLORS[: len(senders)]


def _save(fig, name: str, out_dir: Path) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _short(name: str) -> str:
    """İlk kelimeyi döner — grafik etiketleri için."""
    return name.split()[0]


# ── Grafik 1: Özet Dashboard ─────────────────────────────────────────────────
def plot_summary(df, rt_df, starters, senders, out_dir):
    cols   = _colors(senders)
    fig    = plt.figure(figsize=(16, 10))
    fig.suptitle("WhatsApp Sohbet Analizi — Özet", fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # 1a Pasta: mesaj paylaşımı
    ax1 = fig.add_subplot(gs[0, 0])
    cnt = df["sender"].value_counts().reindex(senders)
    ax1.pie(cnt.values, labels=[_short(s) for s in senders],
            colors=cols, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title("Mesaj Paylaşımı")

    # 1b Yazı vs medya
    ax2 = fig.add_subplot(gs[0, 1])
    cats = ["Yazı", "Medya"]
    for i, s in enumerate(senders):
        sub  = df[df["sender"] == s]
        vals = [sub[~sub["is_media"]].shape[0], sub[sub["is_media"]].shape[0]]
        x    = np.arange(len(cats)) + i * 0.35
        ax2.bar(x, vals, 0.33, color=cols[i], label=_short(s), alpha=0.85)
    ax2.set_xticks(np.arange(len(cats)) + (len(senders) - 1) * 0.175)
    ax2.set_xticklabels(cats)
    ax2.set_title("Yazı vs Medya")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 1c Konuşma başlatma
    ax3 = fig.add_subplot(gs[0, 2])
    st_vals = [starters[s] for s in senders]
    bars = ax3.bar([_short(s) for s in senders], st_vals, color=cols, alpha=0.85, edgecolor="white")
    for b, v in zip(bars, st_vals):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(v),
                 ha="center", va="bottom", fontweight="bold")
    ax3.set_title(f"Konuşma Başlatma (>{CONV_GAP_HOURS}sa boşluk)")
    ax3.set_ylabel("Adet")

    # 1d Medyan yanıt süresi
    ax4 = fig.add_subplot(gs[1, 0])
    if not rt_df.empty:
        med_rt = {s: statistics.median(rt_df[rt_df["responder"] == s]["gap_min"].tolist() or [0])
                  for s in senders}
        vals = [med_rt[s] for s in senders]
        bars = ax4.bar([_short(s) for s in senders], vals, color=cols, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            lbl = f"{v:.0f}dk" if v < 60 else f"{v/60:.1f}sa"
            ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, lbl,
                     ha="center", va="bottom", fontweight="bold")
    ax4.set_title("Medyan Yanıt Süresi")
    ax4.set_ylabel("Dakika")

    # 1e Ort. kelime/mesaj
    ax5 = fig.add_subplot(gs[1, 1])
    txt = df[~df["is_media"]]
    mean_words = {s: txt[txt["sender"] == s]["word_count"].mean() for s in senders}
    vals = [mean_words[s] for s in senders]
    bars = ax5.bar([_short(s) for s in senders], vals, color=cols, alpha=0.85, edgecolor="white")
    for b, v in zip(bars, vals):
        ax5.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{v:.1f}",
                 ha="center", va="bottom", fontweight="bold")
    ax5.set_title("Ortalama Kelime/Mesaj")
    ax5.set_ylabel("Kelime")

    # 1f Geç yanıt sayısı
    ax6 = fig.add_subplot(gs[1, 2])
    if not rt_df.empty:
        late_counts = {s: rt_df[(rt_df["responder"] == s) & rt_df["is_late"]].shape[0] for s in senders}
        vals = [late_counts[s] for s in senders]
        bars = ax6.bar([_short(s) for s in senders], vals, color=cols, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax6.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(v),
                     ha="center", va="bottom", fontweight="bold")
    ax6.set_title(f"Geç Yanıt (>{LATE_REPLY_MIN}dk) Sayısı")
    ax6.set_ylabel("Adet")

    _save(fig, "01_ozet", out_dir)


# ── Grafik 2: Aylık Trend ─────────────────────────────────────────────────────
def plot_monthly(df, senders, out_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly = df.groupby(["month", "sender"]).size().unstack(fill_value=0)
    monthly = monthly.reindex(columns=senders, fill_value=0)
    monthly.index = monthly.index.astype(str)
    monthly.plot(kind="bar", ax=ax, color=_colors(senders), alpha=0.85,
                 edgecolor="white", width=0.7)
    ax.set_title("Aylık Mesaj Trendi", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Mesaj Sayısı")
    ax.set_xticklabels(monthly.index, rotation=45, ha="right")
    ax.legend([_short(s) for s in senders])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, "02_aylik_trend", out_dir)


# ── Grafik 3: Saatlik Isı Haritası ────────────────────────────────────────────
def plot_hourly_heatmap(df, senders, out_dir):
    fig, axes = plt.subplots(1, len(senders), figsize=(7 * len(senders), 4), sharey=True)
    if len(senders) == 1:
        axes = [axes]
    cmaps = ["Blues", "Reds", "Greens", "Oranges"]
    day_labels = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]

    for ax, s, cmap in zip(axes, senders, cmaps):
        heat = (df[df["sender"] == s]
                .groupby(["weekday", "hour"]).size()
                .unstack(fill_value=0)
                .reindex(index=range(7), columns=range(24), fill_value=0))
        sns.heatmap(heat, ax=ax, cmap=cmap, linewidths=0.3, linecolor="white",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(_short(s), fontsize=12)
        ax.set_yticklabels(day_labels, rotation=0, fontsize=9)
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=90, fontsize=8)
        ax.set_xlabel("Saat")
        ax.set_ylabel("Gün")

    fig.suptitle("Mesaj Aktivite Isı Haritası (Gün × Saat)", fontsize=14, fontweight="bold")
    _save(fig, "03_isi_haritasi", out_dir)


# ── Grafik 4: Yanıt Süresi Dağılımı ──────────────────────────────────────────
def plot_response_dist(rt_df, senders, out_dir):
    if rt_df.empty:
        return
    cols = _colors(senders)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    data = [rt_df[rt_df["responder"] == s]["gap_min"].clip(upper=300).tolist() for s in senders]
    data = [d for d in data if d]
    if data:
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(cols[i])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([_short(s) for s in senders[:len(data)]])
    ax.set_ylabel("Yanıt Süresi (dakika, max 300)")
    ax.set_title("Yanıt Süresi Dağılımı (violin)")

    ax2 = axes[1]
    for s, color in zip(senders, cols):
        vals = sorted(rt_df[rt_df["responder"] == s]["gap_min"].tolist())
        if vals:
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax2.plot(vals, cdf, color=color, lw=2, label=_short(s))
    ax2.axvline(LATE_REPLY_MIN, color="gray", ls="--", lw=1, label=f"{LATE_REPLY_MIN}dk eşiği")
    ax2.set_xlim(0, 300)
    ax2.set_xlabel("Yanıt Süresi (dakika)")
    ax2.set_ylabel("Kümülatif Oran")
    ax2.set_title("Yanıt Süresi CDF")
    ax2.legend()

    fig.suptitle("Yanıt Süreleri Analizi", fontsize=14, fontweight="bold")
    _save(fig, "04_yanit_suresi", out_dir)


# ── Grafik 5: Günlük Zaman Serisi ────────────────────────────────────────────
def plot_daily_timeseries(df, senders, out_dir):
    fig, ax = plt.subplots(figsize=(16, 5))
    daily = df.groupby(["date", "sender"]).size().unstack(fill_value=0)
    daily.index = pd.to_datetime(daily.index)

    for s, color in zip(senders, _colors(senders)):
        if s not in daily.columns:
            continue
        roll = daily[s].rolling(7, min_periods=1).mean()
        ax.plot(daily.index, roll, color=color, lw=2, label=f"{_short(s)} (7g ort)")
        ax.fill_between(daily.index, roll, alpha=0.12, color=color)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_ylabel("Günlük Mesaj (7 günlük ort.)")
    ax.set_title("Günlük Mesaj Aktivitesi (7-Günlük Hareketli Ortalama)", fontsize=14)
    ax.legend()
    _save(fig, "05_gunluk_trend", out_dir)


# ── Grafik 6: Haftalık + Saatlik ─────────────────────────────────────────────
def plot_weekly_hourly(df, senders, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]

    ax = axes[0]
    for s, color in zip(senders, _colors(senders)):
        wk = df[df["sender"] == s]["weekday"].value_counts().reindex(range(7), fill_value=0)
        ax.plot(day_names, wk.values, marker="o", color=color, lw=2, label=_short(s))
        ax.fill_between(day_names, wk.values, alpha=0.1, color=color)
    ax.set_title("Haftanın Günlerine Göre Mesaj")
    ax.set_ylabel("Mesaj Sayısı")
    ax.legend()

    ax2 = axes[1]
    for s, color in zip(senders, _colors(senders)):
        hr = df[df["sender"] == s]["hour"].value_counts().reindex(range(24), fill_value=0)
        ax2.plot(range(24), hr.values, marker="o", ms=4, color=color, lw=2, label=_short(s))
        ax2.fill_between(range(24), hr.values, alpha=0.1, color=color)
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax2.set_title("Saate Göre Mesaj Dağılımı")
    ax2.set_ylabel("Mesaj Sayısı")
    ax2.legend()

    fig.suptitle("Zaman Dağılımı Analizi", fontsize=14, fontweight="bold")
    _save(fig, "06_zaman_dagilim", out_dir)


# ── Grafik 7: Yanıt Süresi Zaman Serisi ──────────────────────────────────────
def plot_rt_over_time(rt_df, senders, out_dir):
    if rt_df.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    for s, color in zip(senders, _colors(senders)):
        sub = rt_df[rt_df["responder"] == s].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dt").resample("7D")["gap_min"].median().dropna()
        ax.plot(sub.index, sub.values, color=color, lw=2, label=f"{_short(s)} (7g med)")
        ax.fill_between(sub.index, sub.values, alpha=0.1, color=color)

    ax.axhline(LATE_REPLY_MIN, color="gray", ls="--", lw=1, label=f"{LATE_REPLY_MIN}dk eşiği")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_ylabel("Medyan Yanıt Süresi (dk)")
    ax.set_title("Yanıt Süresinin Zamanla Değişimi (7-Günlük Median)", fontsize=14)
    ax.legend()
    _save(fig, "07_yanit_zaman", out_dir)


# ── Grafik 8: Mesaj Uzunluğu ──────────────────────────────────────────────────
def plot_msg_length(df, senders, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    txt = df[~df["is_media"]]

    for ax, metric, label in zip(axes,
                                  ["word_count", "char_count"],
                                  ["Kelime Sayısı", "Karakter Sayısı"]):
        for s, color in zip(senders, _colors(senders)):
            vals = txt[txt["sender"] == s][metric]
            vals = vals[vals > 0].clip(upper=vals.quantile(0.98))
            ax.hist(vals, bins=40, color=color, alpha=0.55,
                    label=_short(s), edgecolor="white")
        ax.set_xlabel(label)
        ax.set_ylabel("Mesaj Adedi")
        ax.set_title(f"Mesaj {label} Dağılımı")
        ax.legend()

    fig.suptitle("Mesaj Uzunluğu Analizi", fontsize=14, fontweight="bold")
    _save(fig, "08_uzunluk", out_dir)


# ── Grafik 9: En Uzun Sessizlikler ───────────────────────────────────────────
def plot_silences(df, out_dir):
    silences = []
    for i in range(1, len(df)):
        gap_h = (df.iloc[i]["dt"] - df.iloc[i - 1]["dt"]).total_seconds() / 3600
        if gap_h >= CONV_GAP_HOURS:
            silences.append({"start": df.iloc[i - 1]["dt"], "hours": gap_h})
    if not silences:
        return
    sil_df = pd.DataFrame(silences).nlargest(15, "hours")

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(sil_df)))
    labels = [r["start"].strftime("%d.%m.%y %H:%M") for _, r in sil_df.iterrows()]
    bars = ax.barh(range(len(sil_df)), sil_df["hours"].values, color=bar_colors)
    ax.set_yticks(range(len(sil_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Sessizlik Süresi (saat)")
    ax.set_title("En Uzun 15 Sessizlik", fontsize=14)
    for bar, val in zip(bars, sil_df["hours"]):
        lbl = f"{val:.0f}sa" if val < 72 else f"{val/24:.1f}g"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=8)
    _save(fig, "09_sessizlikler", out_dir)


# ── Grafik 10: Geç Yanıt Takvim ──────────────────────────────────────────────
def plot_late_reply_calendar(rt_df, senders, out_dir):
    if rt_df.empty:
        return
    late = rt_df[rt_df["is_late"]].copy()
    if late.empty:
        return
    late["date"] = late["dt"].dt.date
    daily_late = late.groupby(["date", "responder"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(14, 5))
    for s, color in zip(senders, _colors(senders)):
        sub = daily_late[daily_late["responder"] == s]
        ax.scatter(pd.to_datetime(sub["date"]), sub["count"],
                   color=color, s=sub["count"] * 30, alpha=0.6,
                   label=_short(s), edgecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.set_ylabel("Geç Yanıt Adedi")
    ax.set_title(f"Geç Yanıtların Takvim Dağılımı (>{LATE_REPLY_MIN}dk)", fontsize=14)
    ax.legend()
    _save(fig, "10_gec_yanit_takvim", out_dir)


# ════════════════════════════════════════════════════════════════════════════
# 4. PDF RAPORU
# ════════════════════════════════════════════════════════════════════════════
CHART_META = [
    ("01_ozet.png",             "1. Genel Özet",
     "Mesaj paylaşımı, medya/yazı dağılımı, konuşma başlatıcıları, yanıt süreleri ve geç yanıt sayılarına genel bakış."),
    ("02_aylik_trend.png",      "2. Aylık Mesaj Trendi",
     "Her ay gönderilen mesaj sayısının kişiye göre dağılımı."),
    ("03_isi_haritasi.png",     "3. Aktivite Isı Haritası",
     "Haftanın günleri ve saatler bazında mesaj yoğunluğu."),
    ("04_yanit_suresi.png",     "4. Yanıt Süresi Dağılımı",
     "Violin grafiği ile yanıt süresi dağılımı ve kümülatif dağılım fonksiyonu (CDF)."),
    ("05_gunluk_trend.png",     "5. Günlük Mesaj Aktivitesi",
     "7 günlük hareketli ortalama ile günlük mesaj trendi."),
    ("06_zaman_dagilim.png",    "6. Zaman Dağılımı",
     "Haftanın günlerine ve saatlere göre mesaj dağılımı karşılaştırması."),
    ("07_yanit_zaman.png",      "7. Yanıt Süresinin Zamanla Değişimi",
     "Haftalık medyan yanıt sürelerinin zaman içindeki seyri."),
    ("08_uzunluk.png",          "8. Mesaj Uzunluğu Analizi",
     "Kelime ve karakter sayısı bazında mesaj uzunluğu histogramları."),
    ("09_sessizlikler.png",     "9. En Uzun Sessizlikler",
     "Sohbette yaşanan en uzun 15 sessizlik dönemi."),
    ("10_gec_yanit_takvim.png", "10. Geç Yanıt Takvim Dağılımı",
     f"60 dakikayı aşan yanıtların zaman içindeki takvim dağılımı."),
]


def _draw_cover(pdf: PdfPages, senders: list[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(DARK)

    fig.text(0.5, 0.62, "WhatsApp", ha="center", fontsize=48,
             fontweight="bold", color="white", alpha=0.15)
    fig.text(0.5, 0.54, "Sohbet Analizi", ha="center", fontsize=42,
             fontweight="bold", color="white")
    fig.text(0.5, 0.44, "Kapsamlı İstatistiksel Rapor", ha="center",
             fontsize=18, color=ACCENT)

    ax = fig.add_axes([0.15, 0.40, 0.70, 0.002])
    ax.set_facecolor(ACCENT)
    ax.axis("off")

    names = " · ".join(senders)
    fig.text(0.5, 0.33, names, ha="center", fontsize=14, color="white", alpha=0.85)
    fig.text(0.5, 0.26, f"Oluşturulma Tarihi: {datetime.now():%d %B %Y}",
             ha="center", fontsize=12, color="white", alpha=0.7)
    fig.text(0.5, 0.19,
             "10 Grafik  ·  Mesaj Trendleri  ·  Yanıt Süreleri  ·  Aktivite Haritaları",
             ha="center", fontsize=11, color="white", alpha=0.5)

    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  ✓ Kapak sayfası eklendi")


def _draw_image_page(pdf: PdfPages, img_path: Path,
                     title: str, description: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))

    header = fig.add_axes([0, 0.91, 1, 0.09])
    header.set_facecolor(DARK)
    header.axis("off")
    header.text(0.03, 0.5, title, va="center", ha="left",
                fontsize=16, fontweight="bold", color="white",
                transform=header.transAxes)

    img_ax = fig.add_axes([0.01, 0.12, 0.98, 0.78])
    img_ax.imshow(mpimg.imread(str(img_path)), aspect="auto")
    img_ax.axis("off")

    footer = fig.add_axes([0, 0, 1, 0.12])
    footer.set_facecolor("#f0f4f8")
    footer.axis("off")
    footer.text(0.03, 0.55, description, va="center", ha="left",
                fontsize=10, color="#333333",
                transform=footer.transAxes, wrap=True)
    footer.text(0.97, 0.25, f"WhatsApp Sohbet Analizi  |  {datetime.now().year}",
                va="center", ha="right", fontsize=8, color="#999999",
                transform=footer.transAxes)

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {img_path.name} eklendi")


def generate_pdf(out_dir: Path, senders: list[str]) -> None:
    pdf_path = out_dir / "WhatsApp_Analiz_Raporu.pdf"
    print(f"\nPDF raporu oluşturuluyor: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        _draw_cover(pdf, senders)
        for filename, title, desc in CHART_META:
            img = out_dir / filename
            if img.exists():
                _draw_image_page(pdf, img, title, desc)
            else:
                print(f"  ✗ Atlandı (bulunamadı): {filename}")

        meta = pdf.infodict()
        meta["Title"]        = "WhatsApp Sohbet Analizi Raporu"
        meta["Author"]       = "app.py"
        meta["Subject"]      = f"WhatsApp sohbet istatistikleri: {', '.join(senders)}"
        meta["CreationDate"] = datetime.now()

    print(f"\nRapor hazır → {pdf_path.resolve()}")


# ════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE — ana giriş noktası
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    chat_file = sys.argv[1] if len(sys.argv) > 1 else CHAT_FILE
    out_dir   = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    # ── Adım 1: Parse ────────────────────────────────────────────────────────
    print(f"[1/3] Dosya okunuyor: {chat_file}")
    df = parse_chat(chat_file)

    if df.empty:
        print("HATA: Hiç mesaj bulunamadı. Dosya formatını kontrol edin.")
        sys.exit(1)

    # Kişi adları otomatik çekilir
    senders = detect_senders(df)
    print(f"      Kişiler tespit edildi: {', '.join(senders)}")

    # ── Adım 2: Analiz + Grafikler ────────────────────────────────────────────
    rt_df    = compute_response_times(df)
    starters = compute_starters(df)

    print_summary(df, rt_df, starters, senders)

    print("[2/3] Grafikler oluşturuluyor...")
    plot_summary(df, rt_df, starters, senders, out_dir)
    plot_monthly(df, senders, out_dir)
    plot_hourly_heatmap(df, senders, out_dir)
    plot_response_dist(rt_df, senders, out_dir)
    plot_daily_timeseries(df, senders, out_dir)
    plot_weekly_hourly(df, senders, out_dir)
    plot_rt_over_time(rt_df, senders, out_dir)
    plot_msg_length(df, senders, out_dir)
    plot_silences(df, out_dir)
    plot_late_reply_calendar(rt_df, senders, out_dir)

    # ── Adım 3: PDF Raporu ────────────────────────────────────────────────────
    print("[3/3] PDF raporu oluşturuluyor...")
    generate_pdf(out_dir, senders)

    print(f"\nTamamlandı. Çıktılar → {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
