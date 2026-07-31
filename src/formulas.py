"""Typeset the report's formulas as images, in words rather than symbols.

Why images
----------
The report is read as plain Markdown in editors, on GitHub, and in print, and
`$$...$$` renders as literal dollar signs in most of those. A formula that shows
up as raw markup is worse than no formula. Rendering each one to PNG makes it
display identically everywhere.

Why words
---------
The formulas are also rewritten to say what the symbols mean. `lambda* = 2(mu_a -
mu_b) / (var_a - var_b)` requires the reader to hold four subscripted definitions
in mind; "느슨한 쪽 평균 - 엄격한 쪽 평균" requires none. Greek letters are kept
only where the report refers to them again later (lambda, rho), and are always
introduced beside their meaning.

The layout engine
-----------------
Tokens are laid out left to right on a shared baseline. A fraction token stacks
its numerator and denominator around a rule, centred on the wider of the two.
Widths come from the renderer rather than from character counts, so Korean,
Latin and Greek mix without the pieces drifting apart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

FIG_DIR = config.ROOT / "figures"

FONT = "Apple SD Gothic Neo"
# NOTE: this face has no U+2212 MINUS SIGN, so formulas use the ASCII hyphen
INK = "#0b0b0b"
RULE = "#0b0b0b"
SURFACE = "#fcfcfb"
SIZE = 19
DPI = 200

# vertical geometry, in multiples of the base font size
NUM_RISE, DEN_DROP, RULE_PAD = 0.62, 0.72, 0.30


def _text_width(fig, s, size):
    """Measured width of a string in figure pixels."""
    t = fig.text(0, 0, s, fontsize=size, family=FONT)
    fig.canvas.draw()
    w = t.get_window_extent(fig.canvas.get_renderer()).width
    t.remove()
    return w


def render(tokens, name, size=SIZE):
    """Lay out `tokens` on one baseline and save the result.

    A token is either ("t", text) for inline text or ("f", numerator,
    denominator) for a stacked fraction.
    """
    probe = plt.figure(figsize=(1, 1), dpi=DPI)
    gap = _text_width(probe, " ", size)
    widths = []
    for kind, *parts in tokens:
        if kind == "t":
            widths.append(_text_width(probe, parts[0], size))
        else:
            widths.append(max(_text_width(probe, p, size) for p in parts)
                          + 2 * gap)
    plt.close(probe)

    total = sum(widths) + gap * (len(tokens) - 1)
    has_frac = any(k == "f" for k, *_ in tokens)
    height = size * DPI / 72 * (2.5 if has_frac else 1.4)
    fig = plt.figure(figsize=(total / DPI + 0.3, height / DPI), dpi=DPI,
                     facecolor=SURFACE)
    px_x, px_y = fig.get_size_inches() * DPI
    baseline = 0.5
    x = (px_x - total) / 2

    for (kind, *parts), w in zip(tokens, widths):
        cx = (x + w / 2) / px_x
        if kind == "t":
            fig.text(cx, baseline, parts[0], fontsize=size, family=FONT,
                     color=INK, ha="center", va="center")
        else:
            num, den = parts
            unit = size * DPI / 72 / px_y
            fig.text(cx, baseline + NUM_RISE * unit, num, fontsize=size,
                     family=FONT, color=INK, ha="center", va="center")
            fig.text(cx, baseline - DEN_DROP * unit, den, fontsize=size,
                     family=FONT, color=INK, ha="center", va="center")
            half = (w / 2 - RULE_PAD * gap) / px_x
            fig.add_artist(plt.Line2D([cx - half, cx + half],
                                      [baseline, baseline],
                                      color=RULE, linewidth=1.3))
        x += w + gap

    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12,
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.name}")
    return path


FORMULAS = {
    # 1장 — 초과수익률
    "eq1_초과수익률.png": [
        ("t", "초과수익률 ="),
        ("f", "총 상환액 - 대출 원금", "대출 원금"),
        ("t", "-  ( 국채 이자율 × 보유 기간 )"),
    ],
    # 1장 — Sharpe
    "eq2_샤프.png": [
        ("t", "Sharpe ="),
        ("f", "빈티지별 초과수익의 평균", "빈티지별 초과수익의 표준편차"),
    ],
    # 4장 — 평균-분산 만족도
    "eq3_만족도.png": [
        ("t", "만족도  =  평균 수익  -"),
        ("f", "위험회피도 λ", "2"),
        ("t", "×  ( 표준편차 )²"),
    ],
    # 4장 — 무차별 위험회피도
    "eq4_무차별람다.png": [
        ("t", "무차별 위험회피도 λ*  =  2 ×"),
        ("f", "느슨한 쪽 평균  -  엄격한 쪽 평균",
              "느슨한 쪽 분산  -  엄격한 쪽 분산"),
    ],
    # 7장 — 유효표본수
    "eq5_유효표본수.png": [
        ("t", "유효표본수  ≈  관측치 수 ×"),
        ("f", "1 - 자기상관", "1 + 자기상관"),
        ("t", "=  26 ×"),
        ("f", "1 - 0.956", "1 + 0.956"),
        ("t", "≈  0.58"),
    ],
}


def main():
    print("수식 이미지 생성...")
    for name, tokens in FORMULAS.items():
        render(tokens, name)
    print(f"완료 -> {FIG_DIR}")


if __name__ == "__main__":
    main()
