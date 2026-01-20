import numpy as np
import pandas as pd

timestamps = pd.date_range(start='2016-01-01 00:00:00', end='2017-12-31 23:55:00', freq='5T')
n_samples = len(timestamps)

correlations = []

for _ in range(100):
    data_1 = np.random.normal(loc=100, scale=20, size=n_samples)
    data_1 = np.maximum(data_1, 0)

    data_2 = np.random.normal(loc=150, scale=40, size=n_samples)
    data_2 = np.maximum(data_2, 0)

    series_1 = pd.Series(data_1)
    series_2 = pd.Series(data_2)

    spearman_correlation = series_1.corr(series_2, method='spearman')
    correlations.append(spearman_correlation)

count_magnitude_neg_3 = 0
count_magnitude_neg_1 = 0

for correlation in correlations:
    abs_corr = abs(correlation)
    if 0.001 <= abs_corr < 0.01:
        count_magnitude_neg_3 += 1
    if 0.1 <= abs_corr < 1.0:
        count_magnitude_neg_1 += 1

ratio_neg_3 = count_magnitude_neg_3 / len(correlations)
ratio_neg_1 = count_magnitude_neg_1 / len(correlations)
print(ratio_neg_3)
print(ratio_neg_1)