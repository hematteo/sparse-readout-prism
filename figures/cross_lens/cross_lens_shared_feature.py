"""Figure 4 rebuild, v3: diverge-and-reconverge.

Structure = the argument. One frozen state forks into two lenses; the
reported tokens differ; the two decompositions rejoin on one feature.

Quantities, and what each is a share OF (this is where v2 went wrong):
  lens logit for the reported token .... 38.5 / 39.2   (figure-only numbers)
  f23180's share of the FEATURE SUM .... 75% / 63%     (quoted in the prose)
  largest single other feature ......... +1.7 / +1.6   (+1.7 quoted in prose)
The feature sum is not the lens logit -- they differ by the residual
h~^T r_alpha -- so the bar is normalised to its own total and the logit
is kept as a separate annotation. Nothing is drawn as a fraction of a
denominator the source does not state.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text

OUT = ("/private/tmp/claude-501/-Users-m-Desktop-SRP/"
       "04fa9a5d-6e10-44db-81d2-eda38ad16850/scratchpad/fig4/fig4_cross_lens_v3")

PT = 1 / 72
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "text.color": "#1A1A1A",
})
CJK = "SimSong"
S_TOKEN, S_BODY, S_SMALL = 8.5, 7.2, 7.0
S_TOKEN_CJK = 7.6
BLUE, LIGHT, GRAY, EDGE, NEQ = "#2E5E8E", "#D9D9D9", "#5A5A5A", "#BFBFBF", "#A8442E"

COLS = [
    dict(cx=54.0, lens="English-fitted J-lens", token='“London”', cjk=False,
         logit="38.5 logits", share=0.75, pct="75%", other="largest other feature  +1.7"),
    dict(cx=174.0, lens="Chinese-fitted J-lens", token='“伦敦”', cjk=True,
         logit="39.2 logits", share=0.63, pct="63%", other="largest other feature  +1.6"),
]
PROMPT = "中国的首都是北京。英国的首都是 …"

W, H = 226.8, 148.0
XC = 114.0                                   # midpoint of the two columns
BARW, BARH = 84.0, 11.0
Y = dict(prompt=138.0, state=126.0, fork=117.0, lens=104.0, token=90.0,
         logit=79.5, bar=62.0, other=51.0, join=42.0, feat=26.0,
         featsub=15.5, note=4.0)

fig = plt.figure(figsize=(W * PT, H * PT))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
line = dict(lw=0.6, color=GRAY, zorder=2, solid_capstyle="butt")

ax.text(XC, Y["prompt"], PROMPT, fontname=CJK, fontsize=S_SMALL, color=GRAY,
        va="baseline", ha="center")
ax.text(XC, Y["state"], r"one frozen state  $\tilde{h}_{26}$", fontsize=S_BODY,
        va="baseline", ha="center")

# fork: one state -> two lenses
ax.plot([XC, XC], [Y["state"] - 4, Y["fork"]], **line)
ax.plot([COLS[0]["cx"], COLS[1]["cx"]], [Y["fork"]] * 2, **line)
for c in COLS:
    ax.plot([c["cx"]] * 2, [Y["fork"], Y["fork"] - 6], **line)

for c in COLS:
    cx = c["cx"]
    ax.text(cx, Y["lens"], c["lens"], fontsize=S_BODY, color=GRAY,
            va="baseline", ha="center")
    ax.text(cx, Y["token"], c["token"],
            fontsize=S_TOKEN_CJK if c["cjk"] else S_TOKEN,
            va="baseline", ha="center",
            **({"fontname": CJK} if c["cjk"] else {}))
    ax.text(cx, Y["logit"], c["logit"], fontsize=S_SMALL, color=GRAY,
            va="baseline", ha="center")

    x0 = cx - BARW / 2
    ax.add_patch(Rectangle((x0, Y["bar"]), BARW, BARH, facecolor=LIGHT,
                           edgecolor=EDGE, lw=0.4, zorder=3))
    ax.add_patch(Rectangle((x0, Y["bar"]), BARW * c["share"], BARH,
                           facecolor=BLUE, edgecolor="none", zorder=4))
    ax.text(x0 + BARW * c["share"] / 2, Y["bar"] + BARH / 2, c["pct"],
            fontsize=S_SMALL, color="white", va="center", ha="center", zorder=5)
    ax.text(cx, Y["other"], c["other"], fontsize=S_SMALL, color=GRAY,
            va="baseline", ha="center")

# the contrast, stated once where it happens
ax.text(XC, Y["token"] + 1.0, r"$\neq$", fontsize=10, color=NEQ,
        va="baseline", ha="center")

# join: two decompositions -> one feature
for c in COLS:
    ax.plot([c["cx"]] * 2, [Y["other"] - 4, Y["join"]], **line)
ax.plot([COLS[0]["cx"], COLS[1]["cx"]], [Y["join"]] * 2, **line)
ax.plot([XC, XC], [Y["join"], Y["feat"] + 9], **line)

ax.text(XC, Y["feat"], "f23180", fontsize=S_TOKEN, color=BLUE,
        va="baseline", ha="center")
ax.text(XC, Y["featsub"], "dominant under both lenses", fontsize=S_SMALL,
        color=GRAY, va="baseline", ha="center")
ax.text(XC, Y["note"], "bars give f23180\u2019s share of each lens\u2019s feature sum",
        fontsize=S_SMALL, color=GRAY, va="baseline", ha="center")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=400)

fig.canvas.draw()
R = ax.transData.inverted()
for t in fig.findobj(Text):
    if not t.get_text():
        continue
    bb = t.get_window_extent().transformed(R)
    if bb.x0 < -0.5 or bb.x1 > W + 0.5 or bb.y0 < -0.5 or bb.y1 > H + 0.5:
        print(f"OVERFLOW {t.get_text()!r}: x {bb.x0:.1f}..{bb.x1:.1f} y {bb.y0:.1f}..{bb.y1:.1f}")
print(f"ok  {W}x{H}pt")
