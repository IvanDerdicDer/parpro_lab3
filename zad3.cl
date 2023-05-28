__kernel void jacobi_inner(
    __global double* psi_new,
    __global double* psi,
    const int m,
    const int n
) {
    int i = get_global_id(0) + 1;
    for (int j = 1; j <= n; j++) {
        psi_new[i * (m + 2) + j] = 0.25 * (
            psi[(i - 1) * (m + 2) + j]+psi[(i + 1) * (m + 2) + j] + psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]
        );
    }
}


__kernel void delta_sq(
    __global double* new_arr,
    __global double* old_arr,
    __global double* error,
    const int m,
    const int n
) {
    double dsq = 0;
    double tmp = 0;

    int i = get_global_id(0) + 1;

    for(int j = 1; j <= n; j++) {
        tmp = new_arr[i * (m + 2) + j] - old_arr[i * (m + 2) + j];
        dsq += tmp * tmp;
    }

    error[i] = dsq;
}


__kernel void copy_arr(
    __global double* from_arr,
    __global double* to_arr,
    const int m,
    const int n
) {
    int i = get_global_id(0) + 1;

    for(int j = 1; j <= n; j++) {
        to_arr[i * (m + 2) + j] = from_arr[i * (m + 2) + j];
    }
}

