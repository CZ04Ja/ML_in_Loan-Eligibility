# Loan Eligibility: Classification and Validation

A comparison of classification methods for loan approval labels. The archive separates the historical presentation from a later reproducible preprocessing and evaluation pipeline.

## Method
Remove the record identifier from the features; split the sample before fitting preprocessing; impute numeric and categorical columns within the training pipeline; compare a majority-class baseline, logistic regression and random forest. Report accuracy, macro-F1 and the confusion matrix on the same split.

The target is approval status, not subsequent default or repayment. Historical approval decisions may contain policy and selection biases. A random holdout is not an out-of-time or external-bank validation.

## Results and limitations
The archived 2026-10-22 rerun used 614 rows and a 460/154 stratified split. Logistic regression accuracy was 0.8636 and macro-F1 0.8182. These are later audit results, not a replication of the old presentation's reported score. The raw applicant-level file is deliberately not distributed; its provenance and redistribution terms need checking.

Run `python src/reproduce.py --loan-csv /path/to/authorized.csv` to evaluate a local copy. Without that option the script runs a clearly labelled synthetic smoke test. See `results/archived_audit.json` for the earlier audit, and `docs/DATA_POLICY.md` for the boundary.
