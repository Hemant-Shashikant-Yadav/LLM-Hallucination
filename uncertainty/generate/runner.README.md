# uncertainty/generate/runner.py — generating the records

One record per question (PLAN §6): greedy answer, K samples, P(True), verbalised
confidence, probes, cost. Resumable per set.

| Part | What it does |
|---|---|
| `max_tokens_for` | Token limit by answer kind. |
| `greedy_options` | How to ask for the greedy answer under the server's log-prob mode (`raw`, `topk1`, `none`, from the check stage). |
| `binary_prob` | P(first option) from the first answer token's top log-probs; falls back to the reply text. |
| `read_verbal` | The stated 0–100 number. |
| `answer_record` | Greedy and samples in parallel, then self-evaluation and probes (they need the answers). `self_eval` / `probes` off for the lite mode. |
| `generate_set` | Skips finished questions, runs `item_concurrency` at a time, appends each record, runs a gate on the first 20 new records, gzips the finished set, logs tokens/s. |
