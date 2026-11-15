# JEEBench

`data/eval_sets/full/jeebench.jsonl` · 515 records · schema: [`jeebench.schema.json`](../schema/jeebench.schema.json)

> Problems from the IIT JEE Advanced exams in physics, chemistry and maths.

## At a glance

| | |
|---|---|
| Used for | pipeline evaluation (thesis Objectives 3-4) |
| Records | 515 |
| Source | daman1209arora/jeebench (JEE Advanced) ([link](https://huggingface.co/datasets/daman1209arora/jeebench)) |
| Split | test |
| Pinned revision | `d070d1113ce365df05903763a2017ec2d0d56118` |
| Licence | MIT |
| SHA-256 | `6ae852bf68ac5e40eb556a6e64df1941eb524bbc78915c5f4c67c419083008ab` (matches the file) |

## What the dataset is about

JEEBench has 515 problems from the IIT JEE Advanced exams in physics, chemistry and mathematics: single-correct and multiple-correct multiple choice, integer answers and numeric answers. They need long, careful reasoning and domain knowledge.

## Why it is in this project

The hardest set in the pipeline's hard-problem evaluation (`evaluation/hard_eval.py`, set `jeebench`).

## How this file was made

1. Written by `python -m evaluation.export_eval_sets` from `daman1209arora/jeebench` (test split, pinned revision).
2. `kind` maps the question type: MCQ to choice, MCQ(multiple) to multi, Integer and Numeric to value.
3. Option keys are stored in lower case.

## What each record contains

Each line of the file is one JSON object with these fields (the full types are in the schema):

| Field | Type | Meaning | Example |
|---|---|---|---|
| `id` | string | Problem id: 'jee-' + subject + '-' + the dataset's index. | `jee-phy-1` |
| `subject` | string (`chem`, `math`, `phy`) | Subject: phy (physics), chem (chemistry) or math (mathematics). | `phy` |
| `type` | string (`Integer`, `MCQ`, `MCQ(multiple)`, `Numeric`) | Question type as given by JEEBench. | `MCQ` |
| `exam` | string | The exam paper the problem comes from. | `JEE Adv 2016 Paper 1` |
| `question` | string | The problem statement (may contain LaTeX). | `In a historical experiment to determine Planck's constant, …` |
| `kind` | string (`choice`, `multi`, `value`) | Answer kind: choice = one correct option, multi = one or more correct options, value = an integer or numeric answer. | `choice` |
| `key` | string | Gold answer: the option letter(s) in lower case (for example 'b' or 'abd'), or the number. | `b` |

## Statistics

| | |
|---|---:|
| Records | 515 |
| Question, characters (min / median / max) | 107 / 469 / 1,611 |

Exam years: 2016 to 2023 (16 papers).

**Subject**

| Subject | Count | Share |
|---|---:|---:|
| math | 236 | 46% |
| chem | 156 | 30% |
| phy | 123 | 24% |

**Question type**

| Type | Count | Share |
|---|---:|---:|
| MCQ(multiple) | 186 | 36% |
| Numeric | 137 | 27% |
| MCQ | 110 | 21% |
| Integer | 82 | 16% |

## Example record

A real record from the file:

```json
{
  "id": "jee-math-52",
  "subject": "math",
  "type": "Integer",
  "exam": "JEE Adv 2016 Paper 1",
  "question": "What is the total number of distinct $x \\in[0,1]$ for which $\\int_{0}^{x} \\frac{t^{2}}{1+t^{4}} d t=2 x-1$?",
  "kind": "value",
  "key": "1"
}
```

## Caveats

- `key` is stored as text for every question type, so numeric keys must be parsed before comparing.

## Reference

D. Arora, H. G. Singh, Mausam. Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark for Large Language Models. EMNLP 2023.
