# uncertainty/methods/graph.py — spectral uncertainty

Lin, Trivedi & Sun (2024): the samples as a similarity graph.

| Part | What it does |
|---|---|
| `eigv` | Sum of max(0, 1 − λ) over the normalised Laplacian's eigenvalues: a soft count of meanings. |
| `degree` | One minus the mean normalised degree: average dissimilarity. |
| `eccentricity` | Spread of the samples in the spectral embedding (eigenvalues below 0.9). |
