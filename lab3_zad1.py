import pyopencl as cl
import numpy as np

# Definicija broja gradova
N = 100000

# Generiranje slučajnih koordinata gradova
X = np.random.rand(N).astype(np.float32)
Y = np.random.rand(N).astype(np.float32)

# Inicijalizacija izlaznog niza za prosjeke udaljenosti
output = np.zeros(N, dtype=np.float32)

# Inicijalizacija OpenCL konteksta
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Učitavanje i kompilacija OpenCL programa
with open("zad1.cl", "r") as f:
    program_source = f.read()
program = cl.Program(ctx, program_source).build()

# Stvaranje OpenCL memorijskih objekata
X_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X)
Y_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Y)
output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)

# Izvođenje kernela
program.calculate_distances(queue, (N,), None, X_buf, Y_buf, output_buf, np.int32(N))
cl.enqueue_copy(queue, output, output_buf)

# Računanje konačnog prosjeka udaljenosti
average_distance = np.mean(output)

print("Prosjek udaljenosti: ", average_distance)
