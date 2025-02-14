%%writefile matrixvector_mul.cu

#include<stdio.h>

__global__ void matrix_vector(int *a_dev,int *b_dev,int *c_dev, int row_a,int col_a,int row_b,int col_b)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if (row<row_a && col<col_b)
    {
        int sum_arr=0;

        for(int k=0;k<col_a;k++)
        {
            sum_arr+=a_dev[row*col_a+k]*b_dev[k*col_b+col];
        }
        c_dev[row*col_b+col]=sum_arr;
    }


}


int main()
{

    int row_a=4,col_a=6,row_b=6,col_b=1;
    int *a,*b,*c;

    a=(int *)malloc(row_a*col_a*sizeof(int));
    b=(int *)malloc(row_b*col_b*sizeof(int));
    c=(int *)malloc(row_a*col_b*sizeof(int));

    for(int i=0;i<row_a;i++)
    {
        for(int j=0;j<col_a;j++)
        {
            a[i*col_a+j]=1;
        }
    }

    for(int i=0;i<row_b;i++)
    {
        for(int j=0;j<col_b;j++)
        {
            b[i*col_b+j]=1*4;
        }
    }

    // device pointers
    int *d_a,*d_b,*d_c;
    
    cudaMalloc((void **)&d_a,row_a*col_a*sizeof(int));
    cudaMalloc((void **)&d_b,row_b*col_b*sizeof(int));
    cudaMalloc((void **)&d_c,row_a*col_b*sizeof(int));

    cudaMemcpy(d_a,a,row_a*col_a*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,row_b*col_b*sizeof(int),cudaMemcpyHostToDevice);
    
    dim3 gridsize(ceil(col_b/2.0),ceil(row_a/2.0),1);
    dim3 blocksize(2,2,1);

    matrix_vector<<<gridsize,blocksize>>>(d_a,d_b,d_c,row_a,col_a,row_b,col_b);
    
    cudaMemcpy(c,d_c,row_a*col_b*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<row_a;i++)
    {
        for(int j=0;j<col_b;j++)
        {
            printf("%d",c[i*col_b+j]);
        }
        printf("\n");
    }
}

// To run this use: !nvcc matrixvector_mul.cu -o matrixvector_mul and then !./matrixvector_mul
