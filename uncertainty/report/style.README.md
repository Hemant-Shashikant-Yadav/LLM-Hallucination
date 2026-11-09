# uncertainty/report/style.py — figure style

One look for every thesis figure: the dataviz reference palette in fixed order (adjacent
pairs pass the colour-vision checks), a single blue ramp for magnitudes, 2 px lines, hairline
grid. Each figure's numbers are also in a CSV beside it.

| Part | What it does |
|---|---|
| `setup`, `color`, `save` | Matplotlib settings, the n-th series colour, tight save. |
