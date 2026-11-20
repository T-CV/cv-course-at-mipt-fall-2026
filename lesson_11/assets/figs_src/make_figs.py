"""Схемы для 11-2 «Принципы». Запуск: python make_figs.py  → PNG в ../ (lesson_11/assets)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

C = {  # спокойная палитра
    "txt": "#F4B183",  # текст — оранжевый
    "img": "#9DC3E6",  # картинка — голубой
    "ref": "#A9D18E",  # референс — зелёный
    "cond": "#D9D2E9",  # скалярные условия — сиреневый
    "net": "#E7E6E6",  # сеть — серый
    "edge": "#404040",
}


def box(ax, x, y, w, h, text, color, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=C["edge"], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, text=None, fs=9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color=C["edge"], connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fs)


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# 1. Общий пайплайн генерации
def fig_pipeline():
    fig, ax = canvas(12.5, 4.8, (0, 12.5), (-0.4, 4.6))
    box(ax, 0.2, 3.3, 1.8, 0.8, "промпт", "white")
    box(ax, 2.6, 3.3, 2.2, 0.8, "текстовый\nэнкодер", C["txt"])
    box(ax, 0.2, 2.15, 1.8, 0.8, "$t$, guidance", "white")
    box(ax, 2.6, 2.15, 2.2, 0.8, "MLP → вектор $c$", C["cond"])
    box(ax, 0.2, 1.0, 1.8, 0.8, "шум $z$\n($t=1$)", "white")
    box(ax, 2.6, 1.0, 2.2, 0.8, "патчи → токены", C["img"])
    box(ax, 6.4, 0.8, 2.6, 3.5, "трансформер\n(MM-DiT)\n\nпредсказывает\nскорость $v_\\theta$", C["net"], fs=12, bold=True)
    for y in (3.7, 2.55, 1.4):
        arrow(ax, 2.0, y, 2.6, y)
    for y, t in ((3.7, "токены текста"), (2.55, "через adaLN"), (1.4, "токены картинки")):
        arrow(ax, 4.8, y, 6.4, y)
        ax.text(5.6, y + 0.08, t, ha="center", va="bottom", fontsize=9)
    box(ax, 9.7, 2.2, 2.4, 0.8, "шаг Эйлера\n$x \\leftarrow x - \\Delta t\\, v_\\theta$", "white", fs=10)
    arrow(ax, 9.0, 2.6, 9.7, 2.6)
    box(ax, 9.7, 3.6, 2.4, 0.8, "VAE-декодер → картинка", C["img"], fs=10)
    arrow(ax, 10.9, 3.0, 10.9, 3.6)
    ax.text(11.0, 3.2, "в конце ($t=0$)", fontsize=9)
    arrow(ax, 10.9, 2.2, 3.7, 0.95, rad=-0.3)
    ax.text(7.7, -0.3, "новый $x$ → снова в трансформер; повторить N раз, $t$: 1 → 0", ha="center", fontsize=10)
    save(fig, "p_pipeline.png")


# 2. Patchify
def fig_patchify():
    fig, ax = canvas(12, 3.6, (0, 12), (0, 3.6))
    # картинка
    ax.add_patch(plt.Rectangle((0.2, 0.6), 2.4, 2.4, fc="#FFE699", ec=C["edge"]))
    ax.text(1.4, 1.8, "картинка\n1024 × 1024 × 3", ha="center", va="center")
    arrow(ax, 2.7, 1.8, 3.5, 1.8, "VAE\n÷8")
    # латент с сеткой патчей
    x0, y0, s = 3.6, 0.8, 2.0
    ax.add_patch(plt.Rectangle((x0, y0), s, s, fc=C["img"], ec=C["edge"]))
    for k in range(1, 8):
        ax.plot([x0 + k * s / 8] * 2, [y0, y0 + s], color="white", lw=1)
        ax.plot([x0, x0 + s], [y0 + k * s / 8] * 2, color="white", lw=1)
    ax.add_patch(plt.Rectangle((x0, y0 + s - s / 8), s / 8, s / 8, fc="#2F5597", ec="none"))
    ax.text(x0 + s / 2, 0.35, "латент 128 × 128 × 16\nпатч 2 × 2 → 64 числа", ha="center", fontsize=10)
    arrow(ax, 5.8, 1.8, 6.6, 1.8, "Linear\n64 → d")
    # лента токенов
    for k in range(12):
        fc = "#2F5597" if k == 0 else C["img"]
        ax.add_patch(plt.Rectangle((6.7 + k * 0.38, 1.55), 0.34, 0.5, fc=fc, ec=C["edge"], lw=0.8))
    ax.text(6.7 + 12 * 0.38 + 0.1, 1.8, "…", fontsize=16, va="center")
    ax.text(9.2, 1.0, "64 × 64 = 4096 токенов размерности d", ha="center", fontsize=10)
    ax.text(9.2, 2.4, "одна последовательность — как слова в LLM", ha="center", fontsize=10, style="italic")
    save(fig, "p_patchify.png")


# 3. Cross-attention vs joint attention
def fig_attention():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("white")
    n, m = 3, 5  # текст, картинка (условно)

    ax = axes[0]
    ax.set_title("Cross-attention (SD 1.x, SDXL)", fontsize=12)
    M = np.ones((m, n))
    ax.imshow(M, cmap="Oranges", vmin=0, vmax=2.5, extent=(0, n, 0, m))
    ax.set_xticks([n / 2]); ax.set_xticklabels(["K, V: текст (не меняется)"])
    ax.set_yticks([m / 2]); ax.set_yticklabels(["Q: картинка"], rotation=90, va="center")
    ax.text(n / 2, m / 2, "картинка\nчитает текст", ha="center", va="center", fontsize=11)
    ax.set_xlim(-0.2, n + 0.2); ax.set_ylim(-0.2, m + 0.2)
    for s in ax.spines.values(): s.set_visible(False)

    ax = axes[1]
    ax.set_title("Joint attention (MM-DiT)", fontsize=12)
    N = n + m
    ax.add_patch(plt.Rectangle((0, m), n, n, fc=C["txt"], ec="white", lw=2))
    ax.add_patch(plt.Rectangle((n, m), m, n, fc="#F8CBAD", ec="white", lw=2))
    ax.add_patch(plt.Rectangle((0, 0), n, m, fc="#BDD7EE", ec="white", lw=2))
    ax.add_patch(plt.Rectangle((n, 0), m, m, fc=C["img"], ec="white", lw=2))
    ax.text(n / 2, m + n / 2, "текст→\nтекст", ha="center", va="center")
    ax.text(n + m / 2, m + n / 2, "текст читает\nкартинку", ha="center", va="center")
    ax.text(n / 2, m / 2, "картинка\nчитает текст\n(как cross-attn)", ha="center", va="center", fontsize=10)
    ax.text(n + m / 2, m / 2, "картинка→картинка\n(self-attention)", ha="center", va="center")
    ax.set_xlim(-0.2, N + 0.2); ax.set_ylim(-0.2, N + 0.2)
    ax.set_xticks([n / 2, n + m / 2]); ax.set_xticklabels(["K: текст", "K: картинка"])
    ax.set_yticks([m + n / 2, m / 2]); ax.set_yticklabels(["Q: текст", "Q: картинка"], rotation=90, va="center")
    ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.5, -0.02, "одна матрица (n + m) × (n + m): информация идёт в обе стороны, текст обновляется от слоя к слою",
             ha="center", fontsize=11)
    save(fig, "p_attention.png")


# 4. DoubleStream vs SingleStream
def fig_blocks():
    fig, ax = canvas(12, 5.2, (0, 12), (0, 5.2))
    ax.text(2.9, 4.95, "DoubleStream: свои веса, общий attention", ha="center", fontsize=12, fontweight="bold")
    for x, col, name in [(0.4, C["txt"], "текст"), (3.4, C["img"], "картинка")]:
        box(ax, x, 0.2, 2.0, 0.55, name, col)
        box(ax, x, 1.1, 2.0, 0.6, "свои Q, K, V\n(+ QK-norm)", col, fs=10)
        box(ax, x, 3.2, 2.0, 0.6, "свой MLP", col, fs=10)
        box(ax, x, 4.1, 2.0, 0.45, name, col, fs=10)
        arrow(ax, x + 1, 0.75, x + 1, 1.1)
        arrow(ax, x + 1, 1.7, x + 1, 2.1)
        arrow(ax, x + 1, 2.75, x + 1, 3.2)
        arrow(ax, x + 1, 3.8, x + 1, 4.1)
    box(ax, 0.4, 2.1, 5.0, 0.65, "общий attention по [текст | картинка]", C["net"])

    ax.text(9.4, 4.95, "SingleStream: всё общее", ha="center", fontsize=12, fontweight="bold")
    box(ax, 7.4, 0.2, 4.0, 0.55, "[текст | картинка]", "#E2D3C3")
    box(ax, 7.4, 1.1, 4.0, 0.6, "общая проекция (+ QK-norm)", C["net"], fs=10)
    box(ax, 7.4, 2.1, 1.9, 1.7, "attention", C["net"])
    box(ax, 9.5, 2.1, 1.9, 1.7, "MLP\n(параллельно)", C["net"])
    box(ax, 7.4, 4.1, 4.0, 0.45, "[текст | картинка]", "#E2D3C3", fs=10)
    arrow(ax, 9.4, 0.75, 9.4, 1.1)
    arrow(ax, 8.35, 1.7, 8.35, 2.1); arrow(ax, 10.45, 1.7, 10.45, 2.1)
    arrow(ax, 8.35, 3.8, 9.0, 4.1); arrow(ax, 10.45, 3.8, 9.8, 4.1)
    save(fig, "p_blocks.png")


# 5. Число блоков по моделям
def fig_block_counts():
    models = ["FLUX.1", "FLUX.2 [dev]", "Qwen-Image v1", "FLUX.3 Klein"]
    sep = [0, 0, 0, 15]  # по 5 блоков на поток × 3 потока, внимание внутри потока
    dbl = [19, 8, 60, 0]
    sgl = [38, 48, 0, 28]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("white")
    y = np.arange(len(models))[::-1]
    ax.barh(y, sep, color=C["ref"], label="раздельные (attention внутри потока)")
    ax.barh(y, dbl, left=sep, color=C["img"], label="DoubleStream (свои веса, общий attention)")
    ax.barh(y, sgl, left=np.array(sep) + dbl, color="#7F7F7F", label="SingleStream (всё общее)")
    for yi, a, b, c in zip(y, sep, dbl, sgl):
        x = 0
        for v in (a, b, c):
            if v:
                ax.text(x + v / 2, yi, str(v), ha="center", va="center", color="black" if v != c or c == 0 else "white")
            x += v
    ax.set_yticks(y); ax.set_yticklabels(models)
    ax.set_xlabel("число блоков")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9, frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    save(fig, "p_block_counts.png")


# 6. Текстовый энкодер: несколько промежуточных слоёв
def fig_text_encoder():
    fig, ax = canvas(12, 4.6, (0, 12), (0, 4.6))
    ax.text(2.0, 4.35, "раньше: T5 + CLIP", ha="center", fontsize=12, fontweight="bold")
    box(ax, 0.3, 2.6, 1.6, 1.3, "T5", C["txt"])
    box(ax, 2.1, 2.6, 1.6, 1.3, "CLIP", C["txt"])
    ax.text(1.1, 2.2, "все токены\n→ поток текста", ha="center", fontsize=9)
    ax.text(2.9, 2.2, "один pooled-вектор\n→ adaLN", ha="center", fontsize=9)

    ax.text(8.0, 4.35, "сейчас: VLM, скрытые состояния нескольких слоёв", ha="center", fontsize=12, fontweight="bold")
    n = 9
    for k in range(n):
        y = 0.3 + k * 0.42
        hl = k in (2, 5, 8)
        box(ax, 5.0, y, 2.2, 0.34, f"слой {k + 1}", C["txt"] if hl else C["net"], fs=9)
        if hl:
            arrow(ax, 7.2, y + 0.17, 8.4, 2.1)
    box(ax, 8.4, 1.7, 1.5, 0.8, "concat\nпо каналам", "white", fs=10)
    arrow(ax, 9.9, 2.1, 10.3, 2.1)
    box(ax, 10.3, 1.7, 1.5, 0.8, "Linear\n→ d", "white", fs=10)
    ax.text(8.0, 0.0, "ранние слои — ближе к словам, поздние — к смыслу; модель получает оба", ha="center", fontsize=10,
            style="italic")
    save(fig, "p_text_encoder.png")


# 7. RoPE: одинаковое смещение — одинаковый вклад; смена разрешения
def fig_rope():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.set_title("позиция входит в score только через смещение", fontsize=12)
    g = 8
    for i in range(g):
        for j in range(g):
            ax.add_patch(plt.Rectangle((j, i), 0.92, 0.92, fc="#F2F2F2", ec="none"))
    pairs = [((1, 1), (3, 2), "#C55A11"), ((4, 5), (6, 6), "#2F5597")]
    for (a, b, col) in pairs:
        for (x, y) in (a, b):
            ax.add_patch(plt.Rectangle((x, y), 0.92, 0.92, fc=col, ec="none"))
        ax.add_patch(FancyArrowPatch((a[0] + 0.46, a[1] + 0.46), (b[0] + 0.46, b[1] + 0.46),
                                     arrowstyle="-|>", mutation_scale=14, lw=1.6, color="black"))
    ax.text(4, -0.7, "обе пары: смещение (+2 по x, +1 по y) → одинаковый позиционный вклад в $q^\\top k$", ha="center", fontsize=10)
    ax.set_xlim(-0.2, g); ax.set_ylim(-1.1, g); ax.set_aspect("equal"); ax.axis("off")

    ax = axes[1]
    ax.set_title("другое разрешение — те же смещения", fontsize=12, y=1.0)
    ax.add_patch(plt.Rectangle((0, 3.2), 4, 4, fc=C["img"], ec=C["edge"]))
    ax.text(2, 8.0, "обучение 1024 × 1024\nтокены 64 × 64", ha="center", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((4.6, 3.2), 8, 4, fc="#BDD7EE", ec=C["edge"]))
    ax.add_patch(plt.Rectangle((4.6, 3.2), 4, 4, fc="none", ec=C["edge"], ls="--", lw=1.5))
    ax.text(6.6, 3.5, "это видела", ha="center", fontsize=9)
    ax.text(10.6, 3.5, "это нет", ha="center", fontsize=9)
    ax.text(8.6, 8.0, "генерация 2048 × 1024\nтокены 128 × 64", fontsize=10, ha="center", va="center")
    ax.text(6.3, 1.9, "столбцов 64…127 модель не видела,\nно смещение «на 3 токена правее» — видела много раз",
            ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.2, 12.8); ax.set_ylim(1, 9.0); ax.set_aspect("equal"); ax.axis("off")
    save(fig, "p_rope.png")


# 8. Референсы как токены
def fig_references():
    fig, ax = canvas(12, 5.0, (0, 12), (0, 5.0))
    segs = [("текст", C["txt"], 2.0, "шум: —", "RoPE: (0, 0, 0)"),
            ("генерируемая\nкартинка", C["img"], 3.4, "шум: $t$", "RoPE: (0, h, w)"),
            ("референс 1", C["ref"], 2.6, "шум: 0 (чистый)", "RoPE: (1, h, w)"),
            ("референс 2", "#C5E0B4", 2.6, "шум: 0 (чистый)", "RoPE: (2, h, w)")]
    x = 0.3
    for name, col, w, noise, rope in segs:
        box(ax, x, 2.6, w - 0.1, 0.9, name, col)
        ax.text(x + (w - 0.1) / 2, 3.75, noise, ha="center", fontsize=10)
        ax.text(x + (w - 0.1) / 2, 2.3, rope, ha="center", fontsize=10, family="DejaVu Sans Mono")
        x += w
    ax.text(6.0, 4.45, "одна последовательность → обычный joint attention, новых модулей нет", ha="center",
            fontsize=12, fontweight="bold")
    ax.annotate("", xy=(0.3, 1.9), xytext=(10.9, 1.9),
                arrowprops=dict(arrowstyle="<->", color=C["edge"]))
    ax.text(6.0, 1.45, "первая координата RoPE — номер изображения, (h, w) — позиция внутри него.\n"
                       "Пиксель (h, w) референса и тот же (h, w) картинки совпадают по месту, но отличаются номером.",
            ha="center", va="center", fontsize=10)
    ax.text(6.0, 0.4, "лосс считается только по токенам генерируемой картинки", ha="center", fontsize=10, style="italic")
    save(fig, "p_references.png")


if __name__ == "__main__":
    fig_pipeline(); fig_patchify(); fig_attention(); fig_blocks()
    fig_block_counts(); fig_text_encoder(); fig_rope(); fig_references()


# 9. FLUX.3 Klein: две стадии
def fig_flux3():
    fig, ax = canvas(12, 4.4, (0, 12), (0, 4.4))
    streams = [("текст", C["txt"], 3.2), ("картинка ($t$)", C["img"], 2.0), ("референсы ($t=0$)", C["ref"], 0.8)]
    for name, col, y in streams:
        box(ax, 0.2, y, 2.0, 0.7, name, col, fs=10)
        arrow(ax, 2.2, y + 0.35, 2.7, y + 0.35)
        box(ax, 2.7, y, 2.8, 0.7, "5 блоков, attention\nтолько внутри потока", col, fs=9)
        arrow(ax, 5.5, y + 0.35, 6.3, 2.35)
    ax.text(4.1, 4.15, "stage 1: потоки отдельно", ha="center", fontsize=11, fontweight="bold")
    box(ax, 6.3, 1.3, 2.8, 2.1, "28 общих блоков\n[текст | картинка | реф.]\nвсе видят всех", C["net"], fs=10)
    ax.text(7.7, 4.15, "stage 2: вместе", ha="center", fontsize=11, fontweight="bold")
    arrow(ax, 9.1, 2.35, 9.7, 2.35)
    box(ax, 9.7, 1.95, 2.1, 0.8, "только токены\nкартинки → $v_\\theta$", C["img"], fs=10)
    ax.text(6.0, 0.25, "у каждого типа токенов свои scale / shift / gate; веса attention и MLP в stage 2 общие",
            ha="center", fontsize=10, style="italic")
    save(fig, "p_flux3.png")


fig_flux3()
