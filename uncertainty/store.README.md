# uncertainty/store.py — where results live

File layout of `results/` and safe reading and writing. A set being generated is appended
to `<set>.partial.jsonl` one line at a time, so a crash loses at most that line; it is
gzipped when complete.

| Part | What it does |
|---|---|
| `model_dir`, `gen_paths` | Folder of a model; partial and final generation files of a set. |
| `read_jsonl` | Reads `.jsonl` or `.jsonl.gz`, skipping a cut-off last line. |
| `append_jsonl`, `write_jsonl` | Append one record (flushed); write a whole file (gzip by suffix or on request). |
| `repair_partial` | Drops a cut-off last line before appending again. |
| `finalize` | Merges partial and final records, orders them as the question set, gzips, removes the partial file. |
| `load_generations` | Every record of a model and set, finished or not. |
| `read_json`, `write_json` | JSON with an atomic write (temporary file, then move). |
