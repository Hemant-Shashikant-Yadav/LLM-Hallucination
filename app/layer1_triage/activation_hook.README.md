# app/layer1_triage/activation_hook.py — legacy: read a model's hidden states

> **Not used in the current setup** (`semantic_entropy` mode). In `surrogate_probe` mode it
> is only a fallback when `probe_model.SurrogateExtractor` fails to load. Kept for Phase 1.

Loads a small Hugging Face model (TinyLlama by default) and records the hidden states of its
**last third of layers** during one forward pass, using PyTorch forward hooks.

| Lines | What it does |
|---|---|
| 1–10 | Docstring: hooks on the final third of layers; output feeds `probe_model.py`. |
| 24–63 | **`ActivationExtractor`** constructor: model name, which layers (`probe_layer_fraction` 0.67 → last third), device, half precision; storage for hooks and captured tensors. |
| 69–110 | `load_model` / `_load_model_sync`: loads tokenizer and model in a worker thread, then picks the target layer indices. |
| 112–125 | `_get_total_layers`: layer count from the config (names differ by architecture). |
| 131–150 | `_register_hooks`: one forward hook per target layer. |
| 152–178 | `_get_layer_modules`: finds the list of transformer blocks for Llama-, GPT- and NeoX-style models. |
| 180–197 | `_make_hook`: the hook stores a detached copy of that layer's output. |
| 199–203 | `_remove_hooks`. |
| 209–281 | `extract_activations` / `_extract_sync`: tokenise (max 512), run the model with hooks on, stack the captured layers into `[layers, tokens, hidden]`, return an `ActivationBundle`. |
| 287–296 | `unload`: frees model and GPU memory. |
| 304–331 | **`ActivationBundle`**: the stacked tensor plus layer indices, hidden size, token count, model name. |
| 333–358 | Views: `mean_pooled` (average over tokens), `last_token`, `flattened` (last token of every layer as one vector, the classifier input). |
