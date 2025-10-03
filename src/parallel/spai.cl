__kernel void spai_full(__global const float* A,
                        __global float* M,
                        const int n)
{
    int col = get_global_id(0);
    if (col >= n) return;

    // Allocate local arrays (max n = 64 for demo)
    float mat[64*64];
    float b[64];
    float x[64];

    // Copy A into local mat
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            mat[i*n+j] = A[i*n+j];

    // RHS = unit vector e_col
    for(int i=0;i<n;i++) b[i] = 0.0f;
    b[col] = 1.0f;

    // Gaussian elimination (no pivoting)
    for(int k=0;k<n;k++){
        float diag = mat[k*n+k];
        if(fabs(diag)<1e-12f) continue;
        for(int j=k;j<n;j++)
            mat[k*n+j] /= diag;
        b[k] /= diag;

        for(int i=k+1;i<n;i++){
            float factor = mat[i*n+k];
            for(int j=k;j<n;j++)
                mat[i*n+j] -= factor*mat[k*n+j];
            b[i] -= factor*b[k];
        }
    }

    // Back substitution
    for(int i=n-1;i>=0;i--){
        float sum = b[i];
        for(int j=i+1;j<n;j++)
            sum -= mat[i*n+j]*x[j];
        x[i] = sum;
    }

    // Write result to M
    for(int i=0;i<n;i++)
        M[i*n+col] = x[i];
}
