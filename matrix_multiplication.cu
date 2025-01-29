%%writefile matrix_multiplication.cu

// Multiplication of square matrices of same size

// https://colab.research.google.com/drive/1gsQCDPw7DlCXq60bMcsZjKCtN8q74O3A#scrollTo=o8q_1ceBwVL7
  
#include <iostream>


__global__ void matrix_mul(int *a,int *b,int *c,int N)
{

    int row= blockIdx.y*blockDim.y+threadIdx.y;
    int col= blockIdx.x*blockDim.x+threadIdx.x;

    if (row<N && col<N)
    {
      int sum=0;

      for(int j=0;j<N;j++)
      {
        sum+=a[row*N+j]*b[j*N+col];
      }

      c[row*N+col]=sum;
    }

}




int main()

{

  int N=10;

  int *a,*b,*c;

  a=(int *)malloc(N*N*sizeof(int));
  b=(int *)malloc(N*N*sizeof(int));
  c=(int *)malloc(N*N*sizeof(int));

  for(int i=0;i<N;i++)
  {
    for(int j=0;j<N;j++)
    {
      a[i*N+j]=1;
      b[i*N+j]=2;
    }
  }


  int *d_a,*d_b,*d_c;

  cudaMalloc((void **)&d_a,N*N*sizeof(int));
  cudaMalloc((void **)&d_b,N*N*sizeof(int));
  cudaMalloc((void **)&d_c,N*N*sizeof(int));

  cudaMemcpy(d_a,a,N*N*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,N*N*sizeof(int),cudaMemcpyHostToDevice);
  
  dim3 griddim(ceil(N/16.0),ceil(N/16.0),1);
  dim3 blockdim(16,16,1);

  
  matrix_mul<<<griddim,blockdim>>>(d_a,d_b,d_c,N);


  // Wait for GPU to finish
  cudaDeviceSynchronize();
  
  cudaMemcpy(c,d_c,N*N*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i=0;i<N;i++)
  {
    for(int j=0;j<N;j++)
    {
      printf("%d",c[i*N+j]);
      printf(" ");
    }
    printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}
