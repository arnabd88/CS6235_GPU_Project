#include <iostream>
#include <string>
#include <stdio.h>

#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std ;
  template <class Type> __device__ Type Poly(const::vector<double> &a, const Type &x, Type &y)
  {
     size_t k = a.size();
	 Type Z = 0. ;
	 Type x_i = 1. ;
	 Type y_i = 1. ;
	 size_t i ;

	 Z = a[0]*x*x + a[1]*y*y - 3*x*y ;
	 return Z ;
  }

  template <class Type> __global__ void toyCuda(const::vector<double> &a, const Type &x, Type &y)
  {
     int tid = threadIdx.x ;
	 Type Z = Poly(a, x, y);
	 //return Z ;
  }


void cuda_poly(vector<double> a, float* x, float* y) {
  int i;
  float *cuda_x ;
  float *cuda_y ;
  thrust::device_vector<double> *cuda_a = a;

  cout << "Size = " << a.size() << endl ;
  cudaMalloc((float**)&cuda_x, sizeof(float));
  cudaMalloc((float**)&cuda_y, sizeof(float));
  cudaMalloc((vector<double>**)&cuda_a, sizeof(vector<double>)*a.size());

  cudaMemcpy(cuda_x, x, sizeof(float), cudaMemcpyHostToDevice) ;
  cudaMemcpy(cuda_y, y, sizeof(float), cudaMemcpyHostToDevice) ;
  cudaMemcpy(cuda_a, a, sizeof(float)*a.size(), cudaMemcpyHostToDevice);
//
//  toyCuda<<<1,1>>>( cuda_a, cuda_x, cuda_y) ;
//  cudaDeviceSynchronize();
}

int main()
{
  cout << "nvcc can understand cpp" << endl ;
    size_t i ;

	// vector of poly coeff
	size_t k=8 ;
	vector<double> a(k) ;
	float x, y ;
	a[0] = 1 ;
	for(i=1; i< k; i++) a[i] = a[i-1] + 1 ;
	float z ;
	x = 0;
	y = 1.987;
	//z = Poly(a, x, y);
	cuda_poly(a, &x, &y) ;

	//cout << "Z = " << z << endl ;

}
