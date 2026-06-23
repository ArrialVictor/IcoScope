"""Visual themes — palettes for background, edges, coastlines, default colormap."""

THEMES = {
    "Dark":    dict(bg="black", edge="white",   coast="white", grat="#bbbbbb", cbar="white",  cmap="viridis"),
    "Light":  dict(bg="white", edge="#333333", coast="#222",  grat="#666666", cbar="#222",   cmap="viridis"),
    "CB-safe": dict(bg="black", edge="white",   coast="white", grat="#bbbbbb", cbar="white",  cmap="cividis"),
}

# Perceptually-uniform / colorblind-safe colormaps only.
# Sequential first, then diverging, then cyclic.
CMAPS = ["viridis", "cividis", "plasma", "magma", "inferno",
         "RdBu_r", "coolwarm", "twilight"]
