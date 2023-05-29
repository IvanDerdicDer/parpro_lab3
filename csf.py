import numpy as np
from time import time
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

with open("zad3.cl", "r") as f:
    program_source = f.read()
program = cl.Program(ctx, program_source).build()

jacobi_inner = program.jacobi_inner
delta_sq = program.delta_sq
copy_arr = program.copy_arr


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
        psi_temp: cl.Buffer,
        psi: cl.Buffer,
        m: int,
        n: int,
        queue: cl.CommandQueue
) -> None:
    jacobi_inner(
        queue,
        (m, ),
        None,
        psi_temp,
        psi,
        np.int32(m),
        np.int32(n)
    )


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

out_err = np.zeros(m, dtype=np.float64)

out_err_buf = cl.Buffer(
    ctx,
    cl.mem_flags.WRITE_ONLY,
    size=out_err.nbytes
)

psi_temp_buf = cl.Buffer(
    ctx,
    cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=psi_temp
)

psi_buf = cl.Buffer(
    ctx,
    cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=psi
)

start_time = time()

for step in np.arange(1, num_iter + 1, 1):
    jacobi_step(
        psi_temp_buf,
        psi_buf,
        m,
        n,
        queue
    )

    if step == num_iter:
        break

    copy_arr(
        queue,
        (m, ),
        None,
        psi_temp_buf,
        psi_buf,
        np.int32(m),
        np.int32(n)
    )

delta_sq(
    queue,
    (m, ),
    None,
    psi_temp_buf,
    psi_buf,
    out_err_buf,
    np.int32(m),
    np.int32(n)
)
cl.enqueue_copy(queue, out_err, out_err_buf)

error = np.sqrt(np.sum(out_err)) / b_norm

execution_time = time() - start_time
iter_time = execution_time / num_iter

print(f"{execution_time = }\n{error = }\n{iter_time = }")

"""
Paralelno
execution_time = 0.07820248603820801
error = 0.007035327570426907
iter_time = 0.0007820248603820801

Slijedno
execution_time = 53.69549036026001
error = 0.007045131051774809
iter_time = 0.5369549036026001

ubrzanje cca 687x
"""
