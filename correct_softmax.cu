%%writefile softmax.cu

#include<stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void softmax(int *input_mat,double *output_mat,int row_mat,int col_mat)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    extern __shared__ double shared_exp[]; 
    //double *temp=(double *)malloc(row_mat*sizeof(double));

    // Finding the normalizing factor
    
    if(row<row_mat)
    {
        double exp_sum=0;

        for(int j=0;j<col_mat;j++)
        {
            exp_sum+=exp((double)input_mat[row*col_mat+j]);
        }
        shared_exp[row]=exp_sum;
    }
    
    if (row<row_mat && col<col_mat)
    {
        output_mat[row*col_mat+col]=exp((double)input_mat[row*col_mat+col])/shared_exp[row]; 
    }
}


int main()
{
    int rows_matrix=2,col_matrix=3;
    
    int input_size=rows_matrix*col_matrix*sizeof(int);
    int output_size=rows_matrix*col_matrix*sizeof(double);

    int *matrix,*d_matrix;
    double *softmax_res,*d_softmax_res;

    matrix=(int *)malloc(input_size);
    softmax_res = (double *)malloc(output_size);

    for(int i=0;i<rows_matrix;i++)
    {
        for(int j=0;j<col_matrix;j++)
        {
            matrix[i*col_matrix+j]=1;
        }
    }

    cudaMalloc((void **)&d_matrix,input_size);
    cudaMalloc((void **)&d_softmax_res,output_size);

    cudaMemcpy(d_matrix,matrix,input_size,cudaMemcpyHostToDevice);
    
    dim3 griddim(ceil(col_matrix/2.0),ceil(rows_matrix/2.0),1);
    dim3 blockdim(2,2,1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    softmax<<<griddim,blockdim>>>(d_matrix,d_softmax_res,rows_matrix,col_matrix);
    
    
    cudaMemcpy(softmax_res,d_softmax_res,output_size,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    
    for(int i=0;i<rows_matrix;i++)
    {
        for(int j=0;j<col_matrix;j++)
        {
            printf("%lf\t",softmax_res[i*col_matrix+j]);
        }
        printf("\n");
    }

    printf("\nKernel Execution Time: %f ms\n", milliseconds);

    cudaFree(d_matrix);
    cudaFree(d_softmax_res);
    free(matrix);
    free(softmax_res);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}
