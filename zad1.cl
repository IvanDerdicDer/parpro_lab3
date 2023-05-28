__kernel void calculate_distances(__global const float *X,
                                __global const float *Y,
                                __global float *output,
                                const int N)
{
    int i = get_global_id(0);
    float sum = 0.0f;

    for (int j = 0; j < N; j++)
    {
        float dx = X[i] - X[j];
        float dy = Y[i] - Y[j];
        float distance = sqrt(dx * dx + dy * dy);
        sum += distance;
    }

    output[i] = sum / N;
}
