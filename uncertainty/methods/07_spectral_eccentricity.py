"""
Method 07 of 17: Spectral Eccentricity (how far apart the samples lie on the graph's "map").

Family      meaning of the samples (graph methods 05-07)
Reference   Lin, Trivedi & Sun, 2024 [3]
Needs       the samples' pairwise similarity (W, as in method 05)

What it asks
    Place every sample as a point on a small map where similar answers land close together.
    How spread out are the points?

Formula
    L    = I − D^(−1/2) · W · D^(−1/2)                 the normalised Laplacian (method 05)
    V    = the eigenvectors of L whose eigenvalue λ < 0.9, one row per sample (the "map")
    Ecc  = ‖ V − mean(V) ‖_F

Symbols
    V          coordinates of each sample on the map (k columns, k = number of λ below 0.9)
    mean(V)    the centre of the map (average of the rows)
    ‖ · ‖_F    Frobenius norm: square root of the sum of all squared entries, i.e. the total
               distance of the points from the centre

Worked example
    7 "Canberra" + 3 "Sydney" (similarity 0.95 within, 0.02 between)  ->  Ecc = 1.00

    With clean groups Ecc is about √(groups − 1):
    all agree -> 0;  2 groups -> 1.00;  3 groups -> 1.41;  10 groups -> 3.00

Reading the score
    0 = all samples in one place (sure). Higher = spread over more groups (unsure). None with
    fewer than 2 samples or no meaning data.

Code: eccentricity.
"""

from __future__ import annotations

import numpy as np

from uncertainty.methods.common import normalised_laplacian, samples_similarity
from uncertainty.methods.view import View


def eccentricity(v: View) -> float | None:
    w = samples_similarity(v)
    if w is None:
        return None
    lam, vec = np.linalg.eigh(normalised_laplacian(w))
    k = max(1, int((lam < 0.9).sum()))
    emb = vec[:, :k]
    return float(np.linalg.norm(emb - emb.mean(axis=0)))
