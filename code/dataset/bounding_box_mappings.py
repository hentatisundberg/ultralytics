#!/usr/bin/env python

import numpy as np


def xyxy2xywhn(x: np.ndarray, width: int, height: int) -> np.ndarray:
    assert x.ndim == 2
    assert x.shape[1] == 4
    y = np.empty(x.shape)
    y[:, 0] = np.clip((x[:, 0] + x[:, 2]) * 0.5 / width, 0, 1)  # x center
    y[:, 1] = np.clip((x[:, 1] + x[:, 3]) * 0.5 / height, 0, 1)  # y center
    y[:, 2] = np.clip((x[:, 2] - x[:, 0]) / width, 0, 1)  # width
    y[:, 3] = np.clip((x[:, 3] - x[:, 1]) / height, 0, 1)  # height
    return y


def xywhn2xyxy(x: np.ndarray, width: int, height: int) -> np.ndarray:
    assert x.ndim == 2
    assert x.shape[1] == 4
    y = np.empty(x.shape, dtype=int)
    y[:, 0] = np.clip((x[:, 0] - x[:, 2] * 0.5) * width, 0, width - 1)  # x_min
    y[:, 1] = np.clip((x[:, 1] - x[:, 3] * 0.5) * height, 0, height - 1)  # y_min
    y[:, 2] = np.clip((x[:, 0] + x[:, 2] * 0.5) * width, 0, width - 1)  # x_max
    y[:, 3] = np.clip((x[:, 1] + x[:, 3] * 0.5) * height, 0, height - 1)  # y_max
    return y.astype(int)
