# uncertainty/framework/taxonomy.py — the uncertainty taxonomy

Two axes (how unsure; what kind), four types, the action each calls for, and how a label
is assembled (PLAN §9.2–9.3).

| Part | What it does |
|---|---|
| `TYPES`, `DESCRIPTIONS`, `ACTIONS`, `SUBTYPE_ACTIONS` | knowledge → retrieve evidence (Layer 2); reasoning → structured reasoning (Layer 3); aleatoric → clarify, abstain or correct the premise; temporal → fresh retrieval. |
| `recommended_action` | Accept a sure answer unless overconfidence is flagged; else the type's action. |
| `intervention_label` | Evidence only → knowledge; hint only → reasoning; both → mixed; neither → unresolved. |
| `in_scope` | A wrong answer or disagreeing samples: right-and-sure answers have nothing to classify. |
| `assemble` | Label per in-scope answer: human, then metadata, then intervention; else the reason it has none. |
