import numpy as np
import pandas as pd


def synthesize_series(parts, noise_std, rng):
    return parts["trend"] + parts["seasonal"] + rng.normal(0, noise_std, len(parts["trend"]))


def synthesize_frame(parts_by_name, rng):
    names = list(parts_by_name)
    residuals = np.column_stack([parts_by_name[name]["residual"] for name in names])
    noise = rng.multivariate_normal(np.zeros(len(names)), np.cov(residuals, rowvar=False), size=len(residuals))
    columns = {name: parts_by_name[name]["trend"] + parts_by_name[name]["seasonal"] + noise[:, index]
               for index, name in enumerate(names)}
    return pd.DataFrame(columns)
