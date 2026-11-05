# Data

Source: UCI Bank Marketing, `bank-additional-full.csv` (41,188 rows). https://doi.org/10.24432/C5K306

## Cleaning and split (Hiram)

- Removed 12 exact duplicate rows (41,176 left).
- Kept categorical `"unknown"` as its own category.
- `pdays = 999` means never contacted before, so `pdays_never_contacted` was added.
- Stratified 70/15/15 split, `random_state=42`: train 28,822, validation 6,177, test 6,177. Positive rate about 11.3% in each split.
- One-hot encoding and standardization were fit on the training rows only.
- Default matrices exclude `duration` (target leakage). `*_with` matrices keep it for the leakage comparison.
- Test arrays are saved but should stay unused until the final evaluation.

## Files in `processed/`

- `arrays.npz`: `X_train_no`, `X_val_no`, `X_test_no`, `X_train_with`, `X_val_with`, `X_test_with`, `y_train`, `y_val`, `y_test`
- `meta.json`: sizes, columns, seed
- Fitted preprocessors and the majority-class baseline

Load the no-duration matrices for modeling:

```python
import numpy as np
data = np.load("data/processed/arrays.npz")
X_train, y_train = data["X_train_no"], data["y_train"]
X_val, y_val = data["X_val_no"], data["y_val"]
```

Majority baseline ("always no"): accuracy 0.887, validation PR-AUC 0.113, validation ROC-AUC 0.500.
