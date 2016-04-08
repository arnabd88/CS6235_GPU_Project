//---- This file contains a demo of the gpu kernel for aiding
//---- the work-list distribution of the interval queue for the ibba thread

//---- Describe the kernel functionality -----------------
/*---- 1. The IBBA thread works on the given interval set and updates the interval queue for each of its sequential iterations
  ---- 2. When the interval queue size exceeds a certain threshold number(t), the gpu kernel is invoked to reprioritize rest of the queue ( t+1 to N ) while the IBBA thread keeps working on the 0-t elements of the queue.
  ---- 3. The kernel receives as input M intervals of the queue of P dimensions each. Thus, each element of the queue, has p subintervals corresponding to each dimension.
  ---- 4. Each interval, Mi, is evaluated in the following way:
           The Mi interval of p dimensions is broken down in k subintervals along a specific dimension. 
		   A sampling point along the other fixed dimensions is chosen and the function is evaluated for k sample point. The max value of the evaluations forms the priority value along that dimension.
		   Thus it will have a priority value along each of its dimensions after traversing the k sampling points across each of the p-dimensions for that interval. 
		   The Max of these priority values become the priority,Pi, of Mi interval.
		   This evaluation is done for every interval parallely, resulting in new priority values of:
		     priority List of intervals =  P1,P2,...,PM

		   This priority list is then sorted in order to arrange them in decreasing order of priority.
		   The sorted priority list is returned from the kernel to the cpu thread

*/

//----------  Coding the IBBA Thread using Gaol -----------------------------

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <gaol/gaol.h>
#include "helper_cuda.h"
#include "cuda_interval_lib.h"
#include <vector>
#include <queue>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//#include "cpu_interval.h"

using namespace std ;
//-------- Some experiments -------//
//1. Use the Gaol Library and check the access patterns

template <typename T> struct KernelArray
{
  T* _array ;
  int _size ;
} ;

//--- Function to convert device_vector to structure
template <typename T> KernelArray<T> convertToKernel(thrust::device_vector<T>& dvec)
{
   KernelArray<T> kArray ;
   kArray._array = thrust::raw_pointer_cast(&dvec[0]);
   kArray._size  = (int) dvec.size();
   return kArray ;
}

__global__ void fooKernel(KernelArray<int> inArray)
{
    int Pvalue = 0;
    for(int i=0; i<inArray._size; ++i)
	   Pvalue += inArray._array[i] ;
	
}

void sampleCaller() {
  gaol::interval  x(0,2),  y(-2,2), z ;
  thrust::host_vector<interval_gpu<float>>  w ;

  vector<vector<interval_gpu<float>>> k ;
  thrust::device_vector<vector<interval_gpu<float>>> k1 ;

  interval_gpu<int> i(5,11) ;
  interval_gpu<int> j(y.left(),y.right()) ;

  queue<vector<gaol::interval>> Q ; //-- The queue elements are vectors of intervals in n-dimensions

  z = ( 1 - sqr(x))*cos(5*x) ;

  cout << "z = " << z << endl ;
  cout << "x = " << x.right() << endl ;
  cout << "i = " << i.upper() << endl ;
  cout << "j = " << j.upper() << endl ;
  cout << "j = " << j.lower() << endl ;

  //------- Testing thrust --------------
  thrust::device_vector<int> ivec ;
  KernelArray<int> iArray = convertToKernel(ivec) ;

//  thrust::device_vector<interval_gpu<int>> intvec ;
//  KernelArray<int> 
 //   thrust::device_vector<gaol::interval> gint = w ;

  fooKernel<<<1,1>>>(iArray);
    
}

int main(void)
{
  gaol::init() ;

//  printf("%d",i);
  sampleCaller();

  gaol::cleanup();
  return 0 ;

 }


