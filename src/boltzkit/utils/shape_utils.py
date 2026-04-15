import numpy as np
import math


def squeeze_last_dim(x: np.ndarray) -> np.ndarray:
    """
    Squeeze the last dimension of an array if it has size 1.

    Converts arrays of shape (batch, 1) to (batch,), leaves (batch,) unchanged.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (batch,) or (batch, 1).

    Returns
    -------
    np.ndarray
        Flattened array of shape (batch,).
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        return np.squeeze(x, -1)
    elif x.ndim == 1:
        return x
    else:
        raise ValueError(f"Input must have shape (batch,) or (batch, 1), got {x.shape}")


def get_balanced_grid(n: int) -> tuple[int, int]:
    """
    Given n subplots, returns a (rows, cols) tuple for a balanced layout.

    - Tries to make it roughly square.
    - If not square, prefers one row less than columns.
    """
    if n <= 0:
        raise ValueError("Number of subplots must be positive")

    # Start with the integer square root
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # If rows == cols - 1, good; if rows < cols - 1, try to reduce cols
    if rows < cols - 1:
        cols -= 1
        rows = math.ceil(n / cols)

    return rows, cols


if __name__ == "__main__":
    for i in range(1, 15):
        rows, cols = get_balanced_grid(i)
        print(i, rows, cols, rows * cols)
