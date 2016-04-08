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
#define intervalEpsilon 0.001 
#define outputEpsilon   0.001 
#define K  10 // Sampling number

#define CPU_THRESHOLD 4

using namespace std ;

__host__  __device__  float returnMax( float a, float b) { if(a > b) return a ; else   return b ; }
__host__  __device__  float returnMix( float a, float b) { if(a < b) return a ; else   return b ; }
       //--- Returns the dimension which has the largest width
int IntervalWidth( vector<gaol::interval> X ) {
     if(X.size()==0) { printf("Error: Interval list empty!!!" ) ; exit(-1) ; }
     float Width = X[0].width(); int   index = 0 ;
 	 for(int i=0; i<X.size(); i++) { if(X[i].width() > Width) { Width = X[i].width() ; index = i ; } }
 	 return index ;
}

template <typename T> struct KernelArray { T* _array ; int _size ; } ;
template <typename T> KernelArray<T> convertToKernel(thrust::device_vector<T>& dvec)
{ KernelArray<T> kArray ;
   kArray._array = thrust::raw_pointer_cast(&dvec[0]);
   kArray._size  = (int) dvec.size(); return kArray ; } ;


//From script
gaol::interval cpu_Inclusion_Func_expr ( gaol::interval x, gaol::interval y) {
   gaol::interval z ;
   z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}

float cpu_Func_expr ( float x, float y) {
   float z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}
//From script
__device__ float gpu_Func_expr ( float x, float y )
{
   float z ;
   z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}

__global__ void gpuKernel ( KernelArray<interval_gpu<float>> gpuMainQue, KernelArray<float> gpuPriQue, int dimension )
{
   printf("Testing ................\n");
}

//ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread
void gpuHandlerThread ( KernelArray<interval_gpu<float>> gpuMainQue, KernelArray<float> gpuPriQue, int dimension) {
       if(K*dimension > 512) {
	      cout << "Reduce the K value" << endl ;
	   }
	   else {
	       dim3 dimBlock(K, dimension);
		   dim3 dimGrid(gpuPriQue._size);
		   gpuKernel<<<dimGrid,dimBlock>>>(gpuMainQue, gpuPriQue, dimension) ;
		   cudaDeviceSynchronize();
	   }
}

			

int main()  {
   gaol::init();
   omp_set_num_threads(NUM_THREADS);
   int dimension  =  2  ;                 // From scipt  - defined by the problem
   gaol::interval x_0(-5,7), x_1(0,5) ;   // From script - defined by the problem
   //--- Data structure for the gpu ---
   thrust::device_vector<interval_gpu<float>> gpu_interval_list ;
   thrust::device_vector<float>               gpu_interval_priority ;
   thrust::device_vector<float>               fbestTag ;
   KernelArray<interval_gpu<float>>           gpuMainQue ;
   KernelArray<float>                         gpuPriQue  ;             

   //--- Data structure for the cpu ---
   float  fbest = numeric_limits<float>::max();
   vector<gaol::interval> bestbbInt(dimension) ;
   deque<gaol::interval> MainQue ;
   deque<gaol::interval> TempQue ;
   deque<float> MainQue_priority ;
   deque<float> TempQue_priority ;
   deque<thread> ManageThreads   ;
   vector<gaol::interval> MidPoints(dimension) ;
   int addedIntervalSize = 0;               // Holds information for threshold of gpu call

   TempQue.push_back(x_0) ;              // From script - initalise queue with starting intervals
   TempQue.push_back(x_1) ;

   vector<gaol::interval> X(dimension);  // Intervals to be used inside the while loop
   vector<gaol::interval> X1(dimension);  // Intervals to be used inside the while loop
   vector<gaol::interval> X2(dimension);  // Intervals to be used inside the while loop
   vector<vector<gaol::interval>> Xi ;
   gaol::interval FunctionBound ;

   //---- Initialise fbestTag --------
   fbestTag.push_back(numeric_limits<float>::max()) ;

   //---- Get the priority of the starting interval ----
   gaol::interval PriTemp = cpu_Inclusion_Func_expr( (x_0.left() + x_0.right())/2 , (x_1.left() + x_1.right())/2 ) ;
   TempQue_priority.push_back(PriTemp.left()) ;

   while( TempQue.size()!= 0  || MainQue.size()!=0 ) {
       cout << " Idiot!!... Terminate while :)...." << endl ;
       if(ManageThreads.size() != 0) {   // gpu calls to be joined
	       if(ManageThreads.front().joinable()) ManageThreads.front().join() ;
		   ManageThreads.pop_front() ;
		   MainQue.clear() ;
		   for(int i=0; i < gpu_interval_list.size()/dimension; i++) {   // translate gpu return list to gaol
		       for(int j=0; j< dimension; j++) {
			       interval_gpu<float> ij_gpu = gpu_interval_list[i*dimension + j] ;
				   gaol::interval ij(ij_gpu.lower(), ij_gpu.upper());
				   MainQue.push_back(ij) ;
			   }
			   MainQue_priority.push_back(gpu_interval_priority[i]) ;
		   }
	  }
	  if(TempQue.size() != 0) {
	     addedIntervalSize += TempQue_priority.size() ;
	     for(int i=0; i<TempQue.size()/dimension; i++) {             // push the TempQue to the MainQue
	        for(int j=0; j < dimension; j++) {
	           MainQue.push_back(TempQue[i*dimension+j]) ;
	        }
	   	 MainQue_priority.push_back(TempQue_priority[i]) ;
	     }
		 TempQue_priority.clear();
		 TempQue.clear();
	  }
	  if( addedIntervalSize > CPU_THRESHOLD && ManageThreads.size()==0 ) {   // minimum number of new intervals to trigger gpu
              addedIntervalSize = 0 ;            // reset the interval counter
			  for(int i=1; i<MainQue.size()/dimension; i++) {
			     for(int j=0; j<dimension; j++) {
				    interval_gpu<float> ij(MainQue[i*dimension+j].left(), MainQue[i*dimension+j].right()) ;
					gpu_interval_list.push_back(ij) ;
				 }
				 gpu_interval_priority.push_back(MainQue_priority[i]) ;
				 KernelArray<interval_gpu<float>> gpuMainQue = convertToKernel(gpu_interval_list);
				 KernelArray<float> gpuPriQue                = convertToKernel(gpu_interval_priority);
				 
		      }
		      ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread
	   }
       
	   fbest = returnMax(fbest, fbestTag.front());
	   X.clear(); X1.clear(); X2.clear(); Xi.clear();
	   for(int i=0; i<dimension; i++) {
	       X.push_back(MainQue.front()); MainQue.pop_front();
	   }
	   FunctionBound = cpu_Inclusion_Func_expr( X[0], X[1] ) ; // possibly from script
	   if ( FunctionBound.right() < fbest  ||  X[IntervalWidth(X)].width() <= intervalEpsilon || FunctionBound.width() <= outputEpsilon ) {
	       cout << "Get_Next_Element\n" << endl ;
	   }
	   else {
	       for(int i=0; i<dimension; i++) {
		       if(i == IntervalWidth(X)) {
		              gaol::interval a(X[i].left(), X[i].left() + X[i].width()/2 ) ;
		              gaol::interval b(X[i].left() + X[i].width()/2, X[i].right() ) ;
			          X1.push_back(a); 
			          X2.push_back(b);
			   } else {
			         X1.push_back(X[i]); X2.push_back(X[i]) ;
			   }
		   }
		   Xi.push_back(X1); Xi.push_back(X2) ;
		   for(int i=0; i< 2; i++) {
		   //-- The function expression needs manipulation from script
		        float ei = cpu_Func_expr( Xi[i][0].width()/2 + Xi[i][0].left() ,
		                                            Xi[i][1].width()/2 + Xi[i][1].left() );
			    if(ei > fbest) {
				   fbest = ei ;
				   bestbbInt = Xi[i] ;
				}
			    for (int j=0; i< dimension; j++) {
				   TempQue.push_back(Xi[i][j]) ;
				}
				TempQue_priority.push_back(ei) ;
		   }
	   }
  }
}

