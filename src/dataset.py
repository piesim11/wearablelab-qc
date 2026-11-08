import numpy as np

from .signal_io import CHANNELS


MAX_GAP_MS = 20


def make_windows(data, window_size=250, stride=125, classes=(1, 2, 3, 4, 5, 6)):
    """Restituisce X [finestre, 8, campioni] e y [finestre]."""
    if any(
        not isinstance(n, (int, np.integer)) or n < 1
        for n in (window_size, stride)
    ):
        raise ValueError("window_size e stride devono essere interi positivi.")

    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes deve contenere etichette distinte.")

    signals = data[CHANNELS].to_numpy(dtype=np.float32)
    labels = data["class"].to_numpy()
    delta_t = np.diff(data["time"].to_numpy())

    breaks = np.flatnonzero(
        (labels[1:] != labels[:-1])
        | (delta_t <= 0)
        | (delta_t > MAX_GAP_MS)
    ) + 1

    boundaries = np.r_[0, breaks, len(data)]
    label_map = {label: index for index, label in enumerate(classes)}

    windows, targets = [], []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < window_size or labels[start] not in label_map:
            continue

        for i in range(start, end - window_size + 1, stride):
            windows.append(signals[i:i + window_size].T)
            targets.append(label_map[labels[start]])

    if not windows:
        raise ValueError("Nessuna finestra valida: controlla classi e lunghezza.")

    return np.stack(windows), np.asarray(targets, dtype=np.int64)