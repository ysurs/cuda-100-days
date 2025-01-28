// https://colab.research.google.com/drive/1B5zc45c1czGPGyrgALkEKFGy9AlFPWIw#scrollTo=t7wjDOnRtVIM

#include <iostream>

__global__ void matrix_add(int *a, int *b, int *c, int row, int column)
{
  int r_var= blockIdx.y*blockDim.y+threadIdx.y;
  int c_var= blockIdx.x*blockDim.x+threadIdx.x;

  if (r_var<row && c_var<column)
  {
    c[r_var*column+c_var] = a[r_var*column+c_var] + b[r_var*column+c_var];
  }
}



int main()
{
  
  int row = 10;
  int col = 10;
  int *mat_a, *mat_b, *mat_c;

  mat_a = (int *)malloc(row * col * sizeof(int));
  mat_b = (int *)malloc(row * col * sizeof(int));
  mat_c = (int *)malloc(row * col * sizeof(int));

  for (int i = 0; i < row; i++)
  {
    for (int j=0;j<col;j++)
    {
      mat_a[i*col+j] = 1;
      mat_b[i*col+j] = 2;
    }
  }

  for (int i=0;i<row;i++)
  {
    for (int j=0;j<col;j++)
    {
      printf("%d",mat_a[i*col+j]);
    }
    printf("\n");
  }

  for (int i=0;i<row;i++)
  {
    for (int j=0;j<col;j++)
    {
      printf("%d",mat_b[i*col+j]);
    }
    printf("\n");
  }

  int *d_mat_a, *d_mat_b, *d_mat_c;

  cudaMalloc((void **)&d_mat_a, row * col * sizeof(int));
  cudaMalloc((void **)&d_mat_b, row * col * sizeof(int));
  cudaMalloc((void **)&d_mat_c, row * col * sizeof(int));

  cudaMemcpy(d_mat_a, mat_a, row * col * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat_b, mat_b, row * col * sizeof(int), cudaMemcpyHostToDevice);

  // Important to convert the divisor into float to get correct output
  // No of blocks is determined by the block dimensions
  // Block and grid dimensions are not related, it can be anything but we have to make sure grid of threads are covering entire output
  dim3 griddim(ceil(row/16.0),ceil(col/16.0),1);
  dim3 blockdim(32,32,1);

  matrix_add<<<griddim,blockdim>>>(d_mat_a,d_mat_b,d_mat_c,row,col);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  cudaMemcpy(mat_c,d_mat_c, row * col * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0;i<row;i++)
  {
    for (int j=0;j<col;j++)
    {
      printf("%d",mat_c[i*col+j]);
    }
    printf("\n");
  }

  cudaFree(d_mat_a);
  cudaFree(d_mat_b);
  cudaFree(d_mat_c);

  return 0;
}
