%%writefile matrix_multiplication_different_shapes.cu

#include <iostream>


__global__ void matrix_mul(int *a,int *b,int *c,int N,int M,int P)
{

    int row= blockIdx.y*blockDim.y+threadIdx.y;
    int col= blockIdx.x*blockDim.x+threadIdx.x;

    if (row<N && col<P)
    {
      int sum=0;

      for(int j=0;j<M;j++)
      {
        sum+=a[row*M+j]*b[j*P+col];
      }

      c[row*P+col]=sum;
    }

}




int main()

{

  int N=10;
  int M=20;
  int P=30;

  int *a,*b,*c;

  a=(int *)malloc(N*M*sizeof(int));
  b=(int *)malloc(M*P*sizeof(int));
  c=(int *)malloc(N*P*sizeof(int));

  for(int i=0;i<N;i++)
  {
    for(int j=0;j<M;j++)
    {
      a[i*M+j]=1;
    }
  }

  for(int i=0;i<M;i++)
  {
    for(int j=0;j<P;j++)
    {
      b[i*P+j]=2;
    }
  }


  int *d_a,*d_b,*d_c;

  cudaMalloc((void **)&d_a,N*M*sizeof(int));
  cudaMalloc((void **)&d_b,M*P*sizeof(int));
  cudaMalloc((void **)&d_c,N*P*sizeof(int));

  cudaMemcpy(d_a,a,N*M*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,M*P*sizeof(int),cudaMemcpyHostToDevice);
  
  dim3 griddim(ceil(P/16.0),ceil(N/16.0),1);
  dim3 blockdim(16,16,1);

  
  matrix_mul<<<griddim,blockdim>>>(d_a,d_b,d_c,N,M,P);


  // Wait for GPU to finish
  cudaDeviceSynchronize();
  
  cudaMemcpy(c,d_c,N*P*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i=0;i<N;i++)
  {
    for(int j=0;j<P;j++)
    {
      printf("%d",c[i*P+j]);
      printf(" ");
    }
    printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

}
