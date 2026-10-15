# Additional MultiVerS experiments

Extract the original R2_public_reproducibility.zip, then extract this add-on into the same directory. The add-on replaces the table exporter and adds the results for Tables S23 and S24. It contains the frozen single-sentence MultiVerS scores, all fitted combinations, threshold tests and query-level outcomes. Manuscripts and review correspondence are excluded.

The additional protocol was fixed before outcome computation in data/multivers_extension_protocol.json. The original four SciFact partitions are retained. All four combinations select the individual MultiVerS score. At 100 admitted items per 100 held-out queries, both have precision 0.610714. The context comparison uses the same released fever_sci checkpoint, official tokenizer and forward method for full abstracts and individual candidate sentences. The common risk comparison uses fitting-derived threshold quantiles, independent calibration queries and separately declared risk targets.

Reproduce from the saved scores with Python 3.11 and requirements-admission.txt:

```
python src/run_multivers_extension.py --smoke
python src/run_multivers_extension.py
python src/evaluate_multivers_context.py --smoke
python src/evaluate_multivers_context.py
python src/audit_multivers_extension.py --context
python src/export_r2_tables.py --smoke
python src/export_r2_tables.py
```

To regenerate the single-sentence neural scores, also install requirements-neural.txt and obtain the pinned checkpoint using the existing MultiVerS downloader. Run `python src/infer_multivers_context.py --smoke` before `python src/infer_multivers_context.py`. The full-abstract scores are supplied in the original asset. The add-on retains both raw prediction probabilities and candidate scores, with SHA-256 hashes for every member.
