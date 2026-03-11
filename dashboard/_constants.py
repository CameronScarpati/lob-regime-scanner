"""Shared constants for the LOB Regime Scanner dashboard."""

REGIME_NAMES = {0: "Quiet", 1: "Trending", 2: "Toxic"}
REGIME_COLORS = {0: "#00c853", 1: "#448aff", 2: "#ff5252"}

# Plotly layout defaults for a premium dark terminal look
PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0c1016",
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        size=11,
        color="#a0aab4",
    ),
    title_font=dict(size=13, color="#c8d0d8"),
    hoverlabel=dict(
        bgcolor="#1a2332",
        bordercolor="#2d3f52",
        font_size=11,
        font_color="#d4d8df",
    ),
)

# Shared axis styling
AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.04)",
    zerolinecolor="rgba(255,255,255,0.06)",
    tickfont=dict(size=10, color="#6e7681"),
    title_font=dict(size=11, color="#8b949e"),
)
