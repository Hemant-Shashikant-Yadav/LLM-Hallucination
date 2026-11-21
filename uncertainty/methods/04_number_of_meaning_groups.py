"""
Method 04 of 17: Number of meaning groups (NumSet).

Family      meaning of the samples
Reference   Lin, Trivedi & Sun, 2024 [3]
Needs       the samples, grouped by meaning

What it asks
    How many different meanings do the samples have?

Formula
    Groups = number of distinct meaning groups among the m samples

Symbols
    m      number of valid samples
    group  samples that mean the same (an unreadable answer is a group of its own)

Worked example
    7 "Canberra" and 3 "Sydney"  ->  Groups = 2

Reading the score
    1 = all samples agree; m = all differ. Simple but coarse: a 9/1 split and a 5/5 split both
    give 2. None without meaning data.

Code: num_clusters.
"""

from __future__ import annotations

from uncertainty.methods.common import has_meaning
from uncertainty.methods.view import View


def num_clusters(v: View) -> float | None:
    if v.m < 1 or not has_meaning(v):
        return None
    return float(len(set(v.clusters())))
