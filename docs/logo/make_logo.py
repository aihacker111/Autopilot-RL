"""
Generates the LOTUS logo files (SVG, text converted to outlines so they render the same everywhere).

  python docs/logo/make_logo.py --font-dir path/to/BeVietnamPro

The mark: seen from above, a swarm of motorbikes (amber and coral capsules) and the self-driving
car's path (teal, the colour of the car's outline in the simulator) weaving cleanly through it.
"""
import argparse, os
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK_DARK, SUB, SUB_DARK = "#1e2327", "#f0efec", "#596169", "#a3abb0"
TEAL_A, TEAL_B = "#0b8c80", "#35cbbb"


BIKES = [   # (x, y, rotation, scale, colour) in a 120 x 120 box, all clear of the path
    (31, 31, -8, 1.5, "#f2b632"), (89, 33, 8, 1.5, "#f2b632"), (25, 64, -4, 1.4, "#f2b632"), (50, 73, 5, 1.4, "#ef6f4c"),
    (94, 70, 7, 1.4, "#f2b632"), (85, 101, -6, 1.5, "#f2b632"), (27, 98, 4, 1.4, "#ef6f4c"),
]
PATH = "M50 106 C50 88 75 84 73 65 C71 49 58 50 58 36"


def mark(uid, tile="#15181b", ring=None):
    """the mark in a 120 x 120 box: a rounded tile, the swarm, and the car's path ending in an arrow"""
    g = f'''<defs><linearGradient id="{uid}t" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="{TEAL_A}"/><stop offset="1" stop-color="{TEAL_B}"/></linearGradient></defs>
<rect width="120" height="120" rx="28" fill="{tile}"/>'''
    if ring: g += f'<rect x=".75" y=".75" width="118.5" height="118.5" rx="27.3" fill="none" stroke="{ring}" stroke-width="1.5"/>'
    for x, y, r, k, c in BIKES:
        g += f'<rect x="{-3.2 * k:.2f}" y="{-6 * k:.2f}" width="{6.4 * k:.2f}" height="{12 * k:.2f}" rx="{3.2 * k:.2f}" fill="{c}" transform="translate({x} {y}) rotate({r})"/>'
    g += f'<path d="{PATH}" fill="none" stroke="url(#{uid}t)" stroke-width="10" stroke-linecap="round"/>'
    g += f'<path d="M58 14 L71 35 L45 35 Z" fill="{TEAL_B}" stroke="{TEAL_B}" stroke-width="5" stroke-linejoin="round"/>'
    return g


def text_path(font_file, text, size, tracking=0.0):
    """text as one SVG path (y down, baseline at 0); returns (d, width, cap_height)"""
    f = TTFont(font_file); gs = f.getGlyphSet(); cmap = f.getBestCmap(); upm = f["head"].unitsPerEm
    s = size / upm; pen = SVGPathPen(gs); x = 0.0
    for ch in text:
        gn = cmap[ord(ch)]
        gs[gn].draw(TransformPen(pen, (s, 0, 0, -s, x, 0)))
        x += gs[gn].width * s + tracking * size
    x -= tracking * size
    cap = f["OS/2"].sCapHeight * s if hasattr(f["OS/2"], "sCapHeight") else size * .7
    return pen.getCommands(), x, cap


def svg(w, h, body, title="LOTUS"):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}" width="{w:.0f}" height="{h:.0f}" role="img" aria-label="{title}"><title>{title}</title>{body}</svg>\n'


def main(a):
    xb, sb = os.path.join(a.font_dir, "BeVietnamPro-ExtraBold.ttf"), os.path.join(a.font_dir, "BeVietnamPro-SemiBold.ttf")
    out = a.out; os.makedirs(out, exist_ok=True)
    word, ww, cap = text_path(xb, "LOTUS", 100, 0.06)
    tag, tw, tcap = text_path(sb, "Self-driving in Vietnamese traffic", 25.5, 0.01)
    files = {}
    files["lotus-mark.svg"] = svg(120, 120, mark("m"), "LOTUS mark")
    files["lotus-mark-dark.svg"] = svg(120, 120, mark("n", ring="rgba(255,255,255,.14)"), "LOTUS mark")
    files["lotus-icon.svg"] = files["lotus-mark.svg"]   # the mark is already an app tile
    files["lotus-icon-dark.svg"] = files["lotus-mark-dark.svg"]
    # horizontal lockups: mark 120 high, word cap height centred on it
    gap, mh = 28, 120
    base = mh / 2 + cap / 2
    W = mh + gap + ww + 4
    for name, ink, ring in (("lotus-logo.svg", INK, None), ("lotus-logo-dark.svg", INK_DARK, "rgba(255,255,255,.14)")):
        files[name] = svg(W, mh, f'{mark(name[:10].replace("-", ""), ring=ring)}<path transform="translate({mh + gap} {base:.1f})" d="{word}" fill="{ink}"/>')
    # lockup with tagline (README banner)
    base2, H2 = 74, 126   # word baseline, then the tagline 38 px lower; room for its descenders
    W2 = mh + gap + max(ww, tw) + 4
    for name, ink, sub, ring in (("lotus-banner.svg", INK, SUB, None), ("lotus-banner-dark.svg", INK_DARK, SUB_DARK, "rgba(255,255,255,.14)")):
        files[name] = svg(W2, H2, f'<g transform="translate(0 {(H2 - mh) / 2:.1f})">{mark(name[:12].replace("-", ""), ring=ring)}</g>'
                                  f'<path transform="translate({mh + gap} {base2:.1f})" d="{word}" fill="{ink}"/>'
                                  f'<path transform="translate({mh + gap + 3} {base2 + 38:.1f})" d="{tag}" fill="{sub}"/>',
                          "LOTUS: self-driving in Vietnamese traffic")
    # for embedding in HTML (the dark page variant adds a faint edge so the tile stays visible)
    files["lotus-mark-inline.svg"] = f'<svg class="mark" viewBox="0 0 120 120" aria-hidden="true">{mark("lm", ring="var(--logo-ring, transparent)")}</svg>\n'
    for n, s in files.items():
        open(os.path.join(out, n), "w").write(s)
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--font-dir", required=True)
    p.add_argument("--out", default=HERE)
    main(p.parse_args())
