"""Pick black or white cell text by the contrast it actually achieves.

Hand-tuned thresholds per colormap are easy to get wrong: white text ends up
on pale salmon, or black text on dark red.  Instead, ask the colormap what
color the cell will really be, compute its relative luminance, and choose
whichever of black and white gives the larger WCAG contrast ratio.

    contrast(white) = 1.05 / (Y + 0.05)
    contrast(black) = (Y + 0.05) / 0.05

these are equal at Y = sqrt(0.0525) - 0.05 = 0.1791, so white wins below that
and black above it.
"""
import matplotlib
import numpy as np

_SWITCH = np.sqrt(0.0525) - 0.05          # 0.17912...


def _luminance(rgb):
    """WCAG relative luminance of an sRGB triple in [0, 1]."""
    c = np.asarray(rgb[:3], dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(np.dot(lin, (0.2126, 0.7152, 0.0722)))


def text_color_fn(cmap, vmin, vmax):
    """Return f(value) -> 'white' or 'black' for cells drawn with this map."""
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    span = max(vmax - vmin, 1e-12)

    def pick(v):
        if not np.isfinite(v):
            return "gray"
        t = min(max((v - vmin) / span, 0.0), 1.0)     # imshow clips likewise
        return "white" if _luminance(cm(t)) < _SWITCH else "black"

    return pick
