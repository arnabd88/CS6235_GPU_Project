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
//	  vector<interval_gpu<float>> gpu_x ;
//	   #pragma omp parallel for
//	   for(int i=0; i<x.size()/dimension; i++) {
//	        for(int j=0; j<dimension; j++) {
//	             interval_gpu<float> ij(x[i*dimension+j].left(), x[i*dimension+j].right()) ;
//	     		gpu_x.push_back(ij) ;
//	        }
//	   }
//
//----------  Coding the IBBA Thread using Gaol -----------------------------

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gaol/gaol.h>
#include "helper_cuda.h"
#include "cuda_interval_lib.h"
#include <limits>
#include <vector>
#include <queue>
#include <deque>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "omp.h"
#include <thread>
#define NUM_THREADS 4

#define CPU_THRESHOLD 2

//#include "cpu_interval.h"

using namespace std ;
//-------- Some experiments -------//
//1. Use the Gaol Library and check the access patterns

//---- Allocate some global shared memory here for writing by gpu and reading by cpu thread -----

//float fbestTag=0.0 ;

__host__  __device__  float returnMax( float a, float b) { if(a > b) return a ; else   return b ; }
__host__  __device__  float returnMix( float a, float b) { if(a < b) return a ; else   return b ; }

template <typename T> struct KernelArray
{
  T* _array ;
  int _size ;
} ;

template <typename T> KernelArray<T> convertToKernel(thrust::device_vector<T>& dvec)
{
   KernelArray<T> kArray ;
   kArray._array = thrust::raw_pointer_cast(&dvec[0]);
   kArray._size  = (int) dvec.size();
   return kArray ;
} ;

void ibbaThread( deque<gaol::interval>&  X ) {
     cout << X[0].left() << endl ;
	 gaol::interval f(9,90);
	 X[0] = f ;
} ;


__global__ void cudaTest ( KernelArray<interval_gpu<float>> inArray, int dim )
{
    int qsize = inArray._size/dim;
    printf("Qsize= %d\n", qsize);
	for(int i=0; i<qsize; i++) {
	    printf("LL = %f",inArray._array[i].lower()) ;
	}
	interval_gpu<float> gpu_local(7,8);
	inArray._array[0] = gpu_local ;
}
void  cudahandleThread ( KernelArray<interval_gpu<float>> inArray, int dim )
{
   cudaTest<<<1,1>>>(inArray, dim);  //--modifies the device_vector directly
   cudaDeviceSynchronize() ;
}

//  template <class Type> Type Func( const Type &x, const Type &y)
//  {
//     Type z ;
//  
//     Type z = (1 - 
//  }

//--- This comes from script ----
gaol::interval cpu_Func_expr ( gaol::interval x, gaol::interval y)
{
   gaol::interval z ;
   
   z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}

__device__ float gpu_Func_expr ( float x, float y )
{
   float z ;
   z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}

int main()
{
   gaol::init();
   omp_set_num_threads(NUM_THREADS);
   thrust::device_vector<interval_gpu<float>> gpu_x ;
   thrust::device_vector<float> gpu_pri_label ;  //-- priority queue sent to the gpu
   thrust::device_vector<float> fbestTag ; //--updated only by the gpu, read by cpu


   int dimension = 2 ;  //From script
   //float fbestTag=0 ;
   gaol::interval x_0(-5,7), x_1(0,5) ;  //From script
   deque<float> x_pri_label ;    //From script
   deque<gaol::interval> x;      //From script
   vector<gaol::interval> varMidPoints(dimension) ;
   deque<thread> manageThreads ;


   x.push_back(x_0) ; //From script   //adding the default intervals
   x.push_back(x_1) ; //From script

   //---- get the priority of the initial interval set ----
   for(int i=0; i<dimension; i++) {
       gaol::interval temp((x[i].left() + x[i].right())/2);
       varMidPoints[i] = temp ;
   }
   gaol::interval PriTemp = cpu_Func_expr(varMidPoints[0], varMidPoints[1]) ; // How to formalize ?
   x_pri_label.push_back((PriTemp.left() + PriTemp.right())/2) ; //Average: although both left and right should be same
   //---- Copy the current queue to the gpu_interval type list(overhead) -----
   #pragma omp parallel for
   for(int i=0; i<x.size()/dimension; i++) {
       gpu_pri_label.push_back(x_pri_label[i]) ;
       for(int j=0; j<dimension; j++) {
	      interval_gpu<float> ij(x[i*dimension+j].left(), x[i*dimension+j].right());
	      gpu_x.push_back(ij);
	   }
   }
   KernelArray<interval_gpu<float>> iArray = convertToKernel(gpu_x);
   //-- create a handler thread for gpu call
   manageThreads.push_back(thread(cudahandleThread, iArray, dimension));
  // manageThreads.front().join() ;



  //--- start the ibba thread here ---
  float fbest = numeric_limits<float>::max();
  cout << "fbest = " << fbest << endl ;
   while(x.size() != 0) {
      //----- Synchronization point -----
      //-- 1. join the threads, 2. receive the sorted queue from gpu, 3. insert the updated queue from last iteration and merge
      manageThreads.front().join();
      //--- translate the device_vector to the current array set : gpu_x to x
      for(int i=0; i<gpu_x.size()/dimension; i++) {
         for(int j=0; j<dimension; j++) {
    	     interval_gpu<float> ij_gpu = gpu_x[i*dimension+j] ;
    	     gaol::interval ij(ij_gpu.lower(), ij_gpu.upper()) ;
    		 x[i*dimension + j] = ij ;
    	 }
    	 x_pri_label[i] = gpu_pri_label[i]; //The priority
     }
 
        if(fbestTag.size()!=0)
           fbest = returnMax(fbestTag.front(), fbest) ;//MAX from gpu kernel(highest priority value)
      // fbest = Max(fbestTag, fbest) ;
       gaol::interval curr_interval = x.front();
       cout << curr_interval << endl ;
       x.pop_front();
 
   }


      //--- structure of the array of gaol's interval queue ---
	  // x = [x_0, x_1,   x_2, x_3,   x_4,x_5, .....]  guaranteed to be multiple of dimension
	  //--- get the translation of gaol to gpu-interval ---
	 // thrust::device_vector<interval_gpu<float>> gpu_x ;
	   #pragma omp parallel for
	   for(int i=0; i<x.size()/dimension; i++) {
	        for(int j=0; j<dimension; j++) {
	             interval_gpu<float> ij(x[i*dimension+j].left(), x[i*dimension+j].right()) ;
	     		gpu_x.push_back(ij) ;
	        }
	   }

	   //KernelArray<interval_gpu<float>> iArray = convertToKernel(gpu_x);
	   cudaTest<<<1,1>>>(iArray, dimension);
	   cudaDeviceSynchronize();

       interval_gpu<float> temp;
	   temp = gpu_x[0];
       printf("Data-Pass Test = %f\n", temp.upper() );

	  //ibbaThread(x);
      // cout << x[0].left() << endl ;



   gaol::cleanup();
}
