__kernel void calculate_pi(
    __global double* output,
    const int iterations_per_thread,
    const int iteration_count
)
{
    int global_id = get_global_id(0);
    int global_size = iteration_count / iterations_per_thread;

    double sum = 0.0;

    for (int i = 0; i < iterations_per_thread; i++){
        double current = (double)(global_id * iterations_per_thread + 1 + i);
        double h = (current - 0.5) / iteration_count;
        sum += (float)4 / (1 +  h * h) / iteration_count;
    }

    output[global_id] = sum;
}
