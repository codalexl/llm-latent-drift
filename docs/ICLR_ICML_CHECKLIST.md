# ICLR / ICML Submission Checklist

## Reproducibility
- [x] Fixed seeds (default 42) across core workflows.
- [x] Deterministic split helper (70/15/15).
- [x] Environment spec in `requirements.txt`.
- [x] Container spec in `Dockerfile`.
- [x] End-to-end launcher in `experiments/run_all.py`.

## Artifacts
- [x] Trust-region model serialization (`models/trust_region_{model}.pkl`).
- [x] Milestone 2 comparative table (Markdown + LaTeX).
- [x] Milestone 3 drift table (Markdown + LaTeX).
- [x] Drift report JSON with lead-time metrics.
- [x] Figure generation for trajectory and drift plots.

## Baselines
- [x] Simplified BRT-Align baseline.
- [x] Simplified N-GLARE baseline.
- [x] SaP proxy baseline.
- [x] Optional real SaP external integration path (`--real-sap`).

## Metrics
- [x] AUROC
- [x] FPR @ 95% safe coverage
- [x] TPR @ 95% safe coverage
- [x] Exit-time summary
- [x] Prefix-to-failure lead time summary
- [x] Generator-shift AUROC drop (optional if shifted activations provided)

## Documentation
- [x] README with exact commands.
- [x] Sanity check doc updated with current CLI.
- [x] Pipeline commands exposed via `latent-dynamics` CLI.

