# ✅ Fixed: CV Fold Error in ML Pipeline

## Problem
**Error:** `n_splits=5 cannot be greater than the number of members in each class`

This occurred because the `SeverityClassifier` was hardcoded to use 5-fold cross-validation, but the training data didn't have enough samples per class to support 5 folds.

## Solution

### 1. Improved Training Data
Updated the training dataset in `app.py`:
- **30 total samples** (was 20)
- **12 Critical** samples
- **9 Medium** samples  
- **9 Low** samples

This provides better class balance and more samples for CV.

### 2. Adaptive CV Parameter
Modified `SeverityClassifier.train()` in `ml_analyzer.py` to use **adaptive cross-validation**:

```python
# Calculate cross-validation score with adaptive cv parameter
n_samples = len(texts)
n_classes = len(set(labels))
min_samples_per_class = min(Counter(labels).values())

# Use minimum cv=3 for small datasets, cv=5 for larger datasets
cv_folds = min(5, min_samples_per_class, n_samples // n_classes)
cv_folds = max(2, cv_folds)  # At least 2 folds, max 5

cv_scores = cross_val_score(self.classifier, X, labels, cv=cv_folds)
```

**How it works:**
- Calculates the minimum number of samples in any single class
- Uses the smaller of: 5 (default), min samples per class, or samples/classes ratio
- Ensures minimum 2 folds and maximum 5 folds
- Automatically scales to your data

### 3. Added Required Import
Added `from collections import Counter` to `ml_analyzer.py` to support the adaptive CV calculation.

## Result
✅ **All severity levels now work** (Critical, Medium, Low)
✅ **Training completes without errors**
✅ **Cross-validation automatically adjusts to data size**
✅ **Handles both small and large datasets**

## Access the App
**URL:** http://localhost:8502  
(Streamlit picked a new port since 8501 was in use)

Try the Severity Classifier now - it should work perfectly!
