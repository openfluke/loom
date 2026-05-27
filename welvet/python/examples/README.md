# welvet Python examples

Runnable scripts for [../README.md](../README.md). Requires `pip install -e .` from `welvet/python` (and `copy_to_python.sh` for native `.so` / `.dll`).

```bash
python3 examples/run_all.py
```

| Script | Topic |
|--------|--------|
| `01_dense_forward.py` | Volumetric JSON, `forward_polymorphic` |
| `02_morph_and_train.py` | `DType` morph, `train()` with shapes |
| `03_save_reload.py` | `serialize()` / `deserialize()` |
| `04_mha_forward.py` | MHA `[batch, seq, d_model]` |
| `05_dna_compare.py` | `dna()` + `compare_dna()` |
