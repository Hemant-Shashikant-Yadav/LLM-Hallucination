# uncertainty/methods/05_spectral_eigv.py — Spectral EigV (a "soft" count of meaning groups)

Method 05 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples (graph methods 05-07) |
| Reference | Lin, Trivedi & Sun, 2024 [3] |
| Needs | the samples' pairwise similarity |
| Code | eigv. |

## What it asks

Draw the samples as dots joined by lines as strong as their similarity. How many "islands" of similar answers are there, counted softly?

## Formula

```
W_jk = similarity of samples j and k, in [0, 1]
       free text: ( P(entail j -> k) + P(entail k -> j) ) / 2, with W_jj = 1
       maths, options, yes/no: 1 if the read answers match, else 0
D    = diagonal matrix of degrees, D_jj = Σ_k W_jk
L    = I − D^(−1/2) · W · D^(−1/2)              the normalised graph Laplacian
EigV = Σ_k  max(0, 1 − λ_k)                     over the eigenvalues λ_k of L
```

## Symbols

| Symbol | Meaning |
|---|---|
| `W` | the similarity table (matrix) of the samples |
| `D_jj` | degree of sample j: how strongly it is joined to all samples, itself included |
| `I` | the identity matrix (1 on the diagonal, 0 elsewhere) |
| `λ_k` | eigenvalues of L: numbers a computer calculates from L. Each island of tightly joined samples gives one λ close to 0, so max(0, 1 − λ) close to 1. |

## Worked example

```
7 "Canberra" + 3 "Sydney"; similarity 0.95 within a group, 0.02 between the groups
eigenvalues: 0, 0.055, and eight between 0.98 and 0.99
EigV = (1 − 0) + (1 − 0.055) + about 0.07 = 2.02

all agree -> about 1;  5/5 -> 2.04;  5/3/2 -> 2.96;  all 10 differ -> 8.47
```

## Reading the score

Like the number of groups (method 04), but near-duplicates count as "almost one group". Higher = more spread. None with fewer than 2 samples or no meaning data.
