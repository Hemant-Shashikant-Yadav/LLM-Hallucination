# uncertainty/methods/04_number_of_meaning_groups.py — Number of meaning groups (NumSet)

Method 04 of 17. Part of [the 17 methods](README.md); the same text is the file's docstring.

| | |
|---|---|
| Family | meaning of the samples |
| Reference | Lin, Trivedi & Sun, 2024 [3] |
| Needs | the samples, grouped by meaning |
| Code | num_clusters. |

## What it asks

How many different meanings do the samples have?

## Formula

```
Groups = number of distinct meaning groups among the m samples
```

## Symbols

| Symbol | Meaning |
|---|---|
| `m` | number of valid samples |
| `group` | samples that mean the same (an unreadable answer is a group of its own) |

## Worked example

```
7 "Canberra" and 3 "Sydney"  ->  Groups = 2
```

## Reading the score

1 = all samples agree; m = all differ. Simple but coarse: a 9/1 split and a 5/5 split both give 2. None without meaning data.
