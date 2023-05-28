import pyopencl as cl
import numpy as np

iteration_count = 100_000_000
iterations_per_thread = 1000
group_count = iteration_count // iterations_per_thread

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Učitavanje i kompilacija OpenCL programa
with open("zad2.cl", "r") as f:
    program_source = f.read()
program = cl.Program(ctx, program_source).build()

# Stvaranje OpenCL memorijskih objekata
output = np.empty(group_count, dtype=np.float64)
output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)

# Izvođenje kernela
program.calculate_pi(
    queue,
    (group_count,),
    None,
    output_buf,
    np.int32(iterations_per_thread),
    np.int32(iteration_count)
)
cl.enqueue_copy(queue, output, output_buf)

my_pi = np.sum(output)

print(f"{my_pi:.16f} {my_pi - np.pi:.16f}")
