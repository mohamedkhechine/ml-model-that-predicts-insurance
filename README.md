# Insurance Cost Prediction — Linear Regression from Scratch

Predicting individual medical insurance charges using linear regression
implemented from scratch with NumPy (no ML libraries for training).

## Dataset
Medical Cost Personal Dataset — 1,338 records.
Features: age, sex, bmi, children, smoker, region. Target: charges (USD).

## Approach
- Z-score normalization (implemented manually)
- Batch gradient descent with squared-error cost (implemented manually)
- 80/20 train-test split
- Compared against scikit-learn's LinearRegression as verification

## Results

| Model | Features | Test cost | Approx. error |
|---|---|---|---|
| Baseline | age, bmi, children | ... | ~11,400 |
| + smoker | ... | ... | ~6,000 |
| + smoker×bmi | ... | ... | ~5,060 |
| scikit-learn | same | 1.02e7 | ... |

## Key finding
The strongest predictor is not smoking alone but the interaction
smoker × bmi. The fitted model gives smoker a negative weight and
smoker×bmi a large positive one, meaning the cost of smoking scales
with BMI rather than being a flat surcharge.

## What I'd do next
- Add region via one-hot encoding
- Investigate the skewed charges distribution
- Compare against a non-linear model

## Files
- `insurance.ipynb` — full implementation
- `insurance.csv` — dataset
