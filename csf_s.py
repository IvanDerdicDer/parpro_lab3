import numpy as np
from time import time


def boundary_psi(
        psi: np.ndarray[np.float64],
        m: int,
        n: int,
        b: int,
        h: int,
        w: int
) -> np.ndarray[np.float64]:
    temp_arr = np.zeros(len(psi), dtype=np.float64)
    for i in np.arange(b + 1, b + w + 1, 1):
        temp_arr[i * (m + 2)] = np.float64(i - b)

    for i in np.arange(b + w, m + 1, 1):
        temp_arr[i * (m + 2)] = np.float64(w)

    for j in np.arange(1, h + 1, 1):
        temp_arr[(m + 1) * (m + 2) + j] = np.float64(w)

    for j in np.arange(h + 1, h + 2, 1):
        temp_arr[(m + 1) * (m + 2) + j] = np.float64(w - j + h)

    return temp_arr


def jacobi_step(
        psi_temp: np.ndarray[np.float64],
        psi: np.ndarray[np.float64],
        m: int,
        n: int
) -> None:
    for i in np.arange(1, m + 1, 1):
        for j in np.arange(1, n + 1, 1):
            psi_temp[i * (m + 2) + j] = 0.25 * (
                        psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] + psi[i * (m + 2) + j - 1] + psi[
                    i * (m + 2) + j + 1])


def delta_sq(
        new_arr: np.ndarray[np.float64],
        old_arr: np.ndarray[np.float64],
        m: int,
        n: int
) -> np.float64:
    dsq = np.sum(
        np.fromiter(
            (
                np.float_power(
                    new_arr[i * (m + 2) + j] - old_arr[i * (m + 2) + j],
                    2
                )
                for i in np.arange(1, m + 1, 1) for j in np.arange(1, n + 1, 1)
            ),
            dtype=np.float64
        ),
        dtype=np.float64
    )

    return dsq


tolerance = 0

b_base = 10
h_base = 15
w_base = 5
m_base = 32
n_base = 32

irrotational = 1
check_err = 0

scale_factor = 16
num_iter = 100

if tolerance:
    check_err = 1

b = b_base * scale_factor
h = h_base * scale_factor
w = w_base * scale_factor
m = m_base * scale_factor
n = n_base * scale_factor

psi = np.zeros((m + 2) * (n + 2), dtype=np.float64)
psi_temp = np.zeros((m + 2) * (n + 2), dtype=np.float64)

psi = boundary_psi(
    psi,
    m,
    n,
    b,
    h,
    w
)

b_norm = np.sum(
    np.fromiter(
        (np.float_power(psi[i * (m + 2) + j], 2) for i in np.arange(m + 2) for j in np.arange(n + 2)),
        dtype=np.float64
    ),
    dtype=np.float64
)
b_norm = np.sqrt(b_norm)

start_time = time()

error: float = 0.0

for step in np.arange(1, num_iter + 1, 1):
    jacobi_step(
        psi_temp,
        psi,
        m,
        n
    )

    if check_err or step == num_iter:
        error = np.sqrt(delta_sq(
            psi_temp,
            psi,
            m,
            n
        )) / b_norm

    for i in np.arange(1, m + 1, 1):
        for j in np.arange(1, m + 1, 1):
            psi[i * (m + 2) + j] = psi_temp[i * (m + 2) + j]


execution_time = time() - start_time
iter_time = execution_time / num_iter

print(f"{execution_time = }\n{error = }\n{iter_time = }")