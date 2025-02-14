%%writefile softmax.cu

#include<stdio.h>
#include <math.h>

__global__ void softmax(int *input_mat,double *output_mat,int row_mat,int col_mat)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    double *temp=(double *)malloc(row_mat*sizeof(double));

    // Finding the normalizing factor
    for(int i=0;i<row_mat;i++)
    {
        double exp_sum=0;

        for(int j=0;j<col_mat;j++)
        {
            exp_sum+=exp((double)input_mat[i*col_mat+j]);
        }
        temp[i]=exp_sum;
    }

    if (row<row_mat && col<col_mat)
    {
        output_mat[row*col_mat+col]=input_mat[row*col_mat+col]/temp[row]; 
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
    
    dim3 griddim(ceil(col_matrix/2.0),ceil(rows_matrix/2.0));
    dim3 blockdim(2,2,1);
    
    softmax<<<griddim,blockdim>>>(d_matrix,d_softmax_res,rows_matrix,col_matrix);
    
    
    cudaMemcpy(softmax_res,d_softmax_res,output_size,cudaMemcpyDeviceToHost);


    
    for(int i=0;i<rows_matrix;i++)
    {
        for(int j=0;j<col_matrix;j++)
        {
            printf("%lf\t",softmax_res[i*col_matrix+j]);
        }
        printf("\n");
    }

    

}
