"""Shared constants for the LOB Regime Scanner dashboard."""

REGIME_NAMES = {0: "Quiet", 1: "Trending", 2: "Toxic"}

# Refined palette: softer, publication-friendly colours that remain
# distinguishable on dark backgrounds and in print (colour-blind safe).
REGIME_COLORS = {
    0: "#4CAF82",   # muted sage green  (Quiet)
    1: "#5C9CF5",   # soft cornflower   (Trending)
    2: "#EF6C6C",   # muted coral red   (Toxic)
}

# Semi-transparent fills for regime backgrounds
REGIME_FILLS = {
    0: "rgba(76,175,130,0.10)",
    1: "rgba(92,156,245,0.10)",
    2: "rgba(239,108,108,0.10)",
}

# ── Panel descriptions (educational) ──
PANEL_DESCRIPTIONS = {
    "heatmap": (
        "Resting limit-order volume across price levels over time. "
        "Bright regions indicate liquidity clusters; the dotted line "
        "tracks the mid-price. Diamond markers flag large trades "
        "(top 10 percentile by size)."
    ),
    "regime_probs": (
        "Posterior state probabilities from a 3-state Gaussian HMM "
        "trained on order-book features. The transition matrix (right) "
        "shows the estimated probability of switching between regimes "
        "in one time step."
    ),
    "depth_surface": (
        "Three-dimensional view of resting volume (z-axis) across "
        "price offsets from mid (x-axis) and time (y-axis). Surface "
        "colour encodes the prevailing regime state."
    ),
    "diagnostics": (
        "Key microstructure indicators: VPIN (Volume-synchronised "
        "Probability of Informed Trading), OFI (Order Flow Imbalance), "
        "quoted spread, and cumulative PnL from a regime-aware strategy. "
        "Background shading reflects detected regimes."
    ),
}

# ── Plotly typography ──
# Use a professional, highly-readable sans-serif stack.
# "Inter" is the primary web font; fallback to system UI fonts.
_FONT_FAMILY = (
    '"Inter", "Helvetica Neue", Helvetica, Arial, sans-serif'
)

# Plotly layout defaults for a clean, publication-ready dark theme
PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0c1016",
    font=dict(
        family=_FONT_FAMILY,
        size=12,
        color="#b0b8c4",
    ),
    title_font=dict(size=15, color="#e0e4ea", family=_FONT_FAMILY),
    hoverlabel=dict(
        bgcolor="#1a2332",
        bordercolor="#3a4a5c",
        font_size=12,
        font_color="#e0e4ea",
        font_family=_FONT_FAMILY,
    ),
)

# Shared axis styling — clean gridlines, readable tick labels
AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.05)",
    zerolinecolor="rgba(255,255,255,0.08)",
    tickfont=dict(size=11, color="#7a8490", family=_FONT_FAMILY),
    title_font=dict(size=12, color="#98a2ae", family=_FONT_FAMILY),
    gridwidth=1,
)
