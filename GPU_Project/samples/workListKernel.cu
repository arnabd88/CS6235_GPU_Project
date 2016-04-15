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
#include <thrust/sort.h>
#include "omp.h"
#include <thread>
#define NUM_THREADS 4
#define intervalEpsilon 0.001 
#define outputEpsilon   0.001 
#define K  10 // Sampling number
#define DIM 2 // Dimension

#define NEW_INTV_THRESHOLD 10
#define CPU_THRESHOLD 4
#define USE_GPU 1

using namespace std ;

int syncFlag=0 ;

__host__  __device__  float returnMax( float a, float b) { if(a > b) return a ; else   return b ; }
__host__  __device__  float returnMin( float a, float b) { if(a < b) return a ; else   return b ; }
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
   kArray._size  = (int) dvec.size();
   return kArray ; } ;


//From script
gaol::interval cpu_Inclusion_Func_expr ( gaol::interval x, gaol::interval y) {
   gaol::interval z ;
   z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}
//End script


//From script
float cpu_Func_expr ( float x, float y) {
   float z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}
//End script

//From script
__device__ float gpu_Func_expr ( float x, float y )
{
   float z ;
   z = (1 - pow(x,2))*cos(5*y) ;
   return z ;
}
//Endscript

//From script
__device__ float gpu_func_array_expr ( float* var ) {
   float z ;
   z = (1 - pow(var[0],2))*cos(5*var[1]) ;
   return z ;
}
//End script

__global__ void gpuKernel ( KernelArray<interval_gpu<float>> gpuMainQue, KernelArray<float> gpuPriQue, int dimension )
{
   //printf("Testing ................\n");
   __shared__ interval_gpu<float> SharedIntervalList[DIM] ; // Warning thrown due to blank constructor
   __shared__ float RandSampleList[DIM] ;
   __shared__ float SharedIntDimSample[DIM][K];
   float localSampleList[DIM] ;
   int tix = threadIdx.x ; int tiy = threadIdx.y ;
   float chunkSize = 0.0 ;
 //  __syncthreads();

   if(tiy==0) {
     SharedIntervalList[tix] = gpuMainQue._array[blockIdx.x*DIM + tix] ;
 //    printf("Copied Interval = %f from thread= %d\n", SharedIntervalList[tix].lower(), tix);
     RandSampleList[tix] = (SharedIntervalList[tix].lower() + SharedIntervalList[tix].upper())/2 ;
   }
   __syncthreads();

   chunkSize = (SharedIntervalList[tix].upper() - SharedIntervalList[tix].lower())/K ;
   SharedIntDimSample[tix][tiy] = SharedIntervalList[tix].lower() +tiy*chunkSize + chunkSize/2 ; //-- Midpoint

   __syncthreads() ;
   for(int m=0; m<DIM; m++) localSampleList[m] = RandSampleList[m] ;

  // localSampleList = RandSampleList;
   localSampleList[tix] = SharedIntDimSample[tix][tiy] ;  // update the random sample for that thread

   SharedIntDimSample[tix][tiy] = gpu_func_array_expr(localSampleList) ;
   __syncthreads();

  //----- Max reduce  for the values per tix and load in RandSampleList(reuse the allocated memory- size of dimension)
  //----- Do a Max reduce ---
  int size = K ;
  for(int i=ceil((float)K/2) ; i>1; i = ceil((float)i/2)) {
       if(tiy < i && tiy+i<size-1)
	      SharedIntDimSample[tix][tiy] = returnMax(SharedIntDimSample[tix][tiy] ,
		                                           SharedIntDimSample[tix][tiy+i]) ;
	   size = i ;
	      // SharedIntDimSample[tix][tiy] += SharedIntDimSample[tix][tiy + i] ;
  __syncthreads() ;
  }
  if(K > 1 && tiy==0) SharedIntDimSample[tix][0] = returnMax(SharedIntDimSample[tix][0],
                                                   SharedIntDimSample[tix][1]) ;
  __syncthreads() ;

  //----- Max vaue across the dimensions ----
  size = 0;
  for(int i=ceil((float)DIM/2) ; i > 1; i = ceil((float)i/2)) {
      if(tix < i && tix+i<size-1)
	     SharedIntDimSample[tix][0] = returnMax(SharedIntDimSample[tix][0],
		                                        SharedIntDimSample[tix+i][0]) ;
	  size = i ;
	  __syncthreads() ;
  }
  //if(tix==0 && tiy==0) {
  //   if(DIM > 1)  SharedIntDimSample[0][0] = returnMax(SharedIntDimSample[0][0],
  //                                               SharedIntDimSample[1][0]) ;
	 //--copy this priority value to the global memory
  //}
  if(tiy==0)
     if(DIM > 1)gpuPriQue._array[blockIdx.x*DIM + tix] = returnMax(SharedIntDimSample[0][0], 
	                                                               SharedIntDimSample[1][0]);
	 else
	     gpuPriQue._array[blockIdx.x*DIM + tix] = SharedIntDimSample[0][0] ;
//	 gpuPriQue._array[blockIdx.x*DIM + tix] = SharedIntDimSample[0][0] ;
  

  //----- The RandSampleList becomes the priority label array of SharedIntervalList -----
  //----- Sort these and terminate

   //---- Sorting done with a thrust call at the host(execution will still be on device)




   __syncthreads();
}

//ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread
void gpuHandlerThread ( KernelArray<interval_gpu<float>> gpuMainQue, KernelArray<float> gpuPriQue, int dimension) {
	   printf("Debug: 7: Got called...\n");
       if(K*dimension > 512) {
	      cout << "Reduce the K value" << endl ;
	   }
	   else {
	       //dim3 dimBlock(K, dimension);
		   dim3 dimBlock(dimension, K);
		   dim3 dimGrid(gpuPriQue._size);
		   gpuKernel<<<dimGrid,dimBlock>>>(gpuMainQue, gpuPriQue, dimension) ;
		   cudaDeviceSynchronize();
		   //thrust::device_vector<float> a_temp = *gpuPriQue._array ;
		  // thrust::sort(gpuPriQue._array[0], gpuPriQue._array.end[4]) ;
	   }

	   syncFlag = 1 ;
}

   //From script	
   int dimension  =  2  ;                 // From scipt  - defined by the problem
   gaol::interval x_0(0,5), x_1(-7,5) ;   // From script - defined by the problem
   //End script

int main()  {
   gaol::init();
   omp_set_num_threads(NUM_THREADS);
   //--- Data structure for the gpu ---
   thrust::device_vector<interval_gpu<float>> gpu_interval_list ;
   thrust::device_vector<float>               gpu_interval_priority ;
   thrust::device_vector<float>               fbestTag ;
   KernelArray<interval_gpu<float>>           gpuMainQue ;
   KernelArray<float>                         gpuPriQue  ;             

   //--- Data structure for the cpu ---
   float  fbest = numeric_limits<float>::min();
   vector<gaol::interval> bestbbInt(dimension) ;
   deque<gaol::interval> MainQue ;
   deque<gaol::interval> TempQue ;
   deque<float> MainQue_priority ;
   deque<float> TempQue_priority ;
   deque<thread> ManageThreads   ;
   vector<gaol::interval> MidPoints(dimension) ;
   int addedIntervalSize = 0;               // Holds information for threshold of gpu call
   int count=0;

   //From script
   TempQue.push_back(x_0) ;              // From script - initalise queue with starting intervals
   TempQue.push_back(x_1) ;
   //End script

   vector<gaol::interval> X(dimension);  // Intervals to be used inside the while loop
   vector<gaol::interval> X1(dimension);  // Intervals to be used inside the while loop
   vector<gaol::interval> X2(dimension);  // Intervals to be used inside the while loop
   vector<vector<gaol::interval>> Xi ;
   gaol::interval FunctionBound ;

   //---- Initialise fbestTag --------
   fbestTag.push_back(numeric_limits<float>::min()) ;

   //---- Get the priority of the starting interval ----
   //From script
   gaol::interval PriTemp = cpu_Inclusion_Func_expr( (x_0.left() + x_0.right())/2 , (x_1.left() + x_1.right())/2 ) ;
   //End script
   for(int i=0; i<dimension; i++) TempQue_priority.push_back(PriTemp.left()) ;

  int loop_Counter = 0 ;
   //-- ManageThreads commented until gpu comes alive
   while( (int)TempQue_priority.size()> 0  || (int)MainQue_priority.size()>0 || (int)ManageThreads.size()>0) {
       // printf("Start: MainQue_priority = %d \n", (int)MainQue_priority.size());
       // printf("Start: TempQue_priority = %d \n", (int)TempQue_priority.size());
       // printf("Start: ManageThreads = %d \n", (int)ManageThreads.size());
	//   printf("Debug: 6: Reached here: Temp_Queue_Size = %lu, Main_Queue_size = %lu\n", TempQue.size(), MainQue.size());
     //  cout << " Idiot!!... Terminate while :)...." << endl ;
       if(USE_GPU==1 && syncFlag == 1 && ManageThreads.size() != 0 && count>=CPU_THRESHOLD) {   // gpu calls to be joined
	       if(ManageThreads.front().joinable()) ManageThreads.front().join() ;
		   loop_Counter++ ;
		 //  if(loop_Counter==1) {
		 cout << "After the re-written priority from gpu " << endl ;
		 for(int i=0; i<(int)gpu_interval_priority.size(); i++) {
		   interval_gpu<float> temp = gpu_interval_list[i] ;
		   cout << temp.lower() << " : " << temp.upper() << " -- " << gpu_interval_priority[i] << endl  ;
		   }
		      cout << "GPU_QUEUE_SIZE = " << (int)gpu_interval_priority.size() << endl ;
		  // }
		   thrust::stable_sort_by_key(gpu_interval_priority.begin(), gpu_interval_priority.end(), gpu_interval_list.begin()) ;
		 cout << "After Sorting " << endl ;
		 for(int i=0; i<(int)gpu_interval_priority.size(); i++) {
		   interval_gpu<float> temp = gpu_interval_list[i] ;
		   cout << temp.lower() << " : " << temp.upper() << " -- " << gpu_interval_priority[i] << endl  ;
		  }
		   syncFlag = 0 ;
		   count = 0 ;
		   ManageThreads.pop_front() ;
		   //MainQue.clear() ;
		   for(int i=0; i < (int)gpu_interval_list.size()/dimension; i++) {   // translate gpu return list to gaol
		       for(int j=dimension-1; j>=0; j--) {
			       interval_gpu<float> ij_gpu = gpu_interval_list[i*dimension + j] ;
				   gaol::interval ij(ij_gpu.lower(), ij_gpu.upper());
				   MainQue.push_front(ij) ;
			       MainQue_priority.push_front(gpu_interval_priority[i*dimension + j]) ;
			   }
			   //MainQue_priority.push_front(gpu_interval_priority[i]) ;
		   }
	  }
	  if((int)TempQue.size() != 0) {
		// cout << " TempQue-Size = " << TempQue_priority.size() << endl ;
	     for(int i=0; i<(int)TempQue.size()/dimension; i++) {             // push the TempQue to the MainQue
	        for(int j=0; j < dimension; j++) {
	           MainQue.push_back(TempQue[i*dimension+j]) ;
	   	       MainQue_priority.push_back(TempQue_priority[i*dimension+j]) ;
	        }
	   	 //MainQue_priority.push_back(TempQue_priority[i]) ;
         //printf("Update-2: MainQue_priority = %d \n", (int)MainQue_priority.size());
	     }
		 TempQue_priority.clear();
		 TempQue.clear();
	  }
	  if(USE_GPU==1 && (int)MainQue_priority.size()/dimension - CPU_THRESHOLD > NEW_INTV_THRESHOLD && ManageThreads.size()==0 ) {   // minimum number of new intervals to trigger gpu
			  gpu_interval_list.clear();
			  gpu_interval_priority.clear();
			    for(int i=CPU_THRESHOLD; i<(int)MainQue.size()/dimension; i++) {
			       for(int j=0; j<dimension; j++) {
			          interval_gpu<float> ij(MainQue[i*dimension+j].left(), MainQue[i*dimension+j].right()) ;
			      	gpu_interval_list.push_back(ij) ;
			      	//cout << " Enter here " << gpu_interval_list.size() << endl ;
			        gpu_interval_priority.push_back(MainQue_priority[i*dimension+j]) ;
			       }
			       //gpu_interval_priority.push_back(MainQue_priority[i]) ;
		        }
		        cout << "Starting  New GPU call of size " << gpu_interval_priority.size() << endl ;
		        for(int i=0; i<(int)gpu_interval_priority.size(); i++) {
				  interval_gpu<float> temp = gpu_interval_list[i] ;
		          cout << temp.lower() << " : " << temp.upper() << " -- " << gpu_interval_priority[i] << endl  ;
				}
		       		//--- Clear the intervals that has been provided to the gpu ---
				MainQue.erase(MainQue.begin() + (CPU_THRESHOLD)*dimension , MainQue.end()) ;
				MainQue_priority.erase(MainQue_priority.begin() + (CPU_THRESHOLD)*dimension, MainQue_priority.end()) ;
				 /* KernelArray<interval_gpu<float>>*/ gpuMainQue = convertToKernel(gpu_interval_list);
				 /* KernelArray<float>*/ gpuPriQue                = convertToKernel(gpu_interval_priority);
		      ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread
	   }
       
	   fbest = returnMax(fbest, fbestTag.front());
	   X.clear(); X1.clear(); X2.clear(); Xi.clear();
	   //printf("MainQueueSize = %lu, TempQueueSize = %lu\n", MainQue_priority.size(), TempQue.size());
	   for(int i=0; i<dimension; i++) {
	       X.push_back(MainQue.front()); MainQue.pop_front(); 
	       MainQue_priority.pop_front();
	   }
	  // MainQue_priority.pop_front();
	   count++ ;
	   //From script
	   FunctionBound = cpu_Inclusion_Func_expr( X[0], X[1] ) ; // possibly from script
	   //End script
	   if ( FunctionBound.right() < fbest  ||  X[IntervalWidth(X)].width() <= intervalEpsilon || FunctionBound.width() <= outputEpsilon ) {
	        //printf("GetNextElement\n");
			//cout << "Current-Size = " << (int)MainQue_priority.size()/dimension << endl ;
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
		        //From script
		        float ei = cpu_Func_expr( Xi[i][0].width()/2 + Xi[i][0].left() ,
		                                            Xi[i][1].width()/2 + Xi[i][1].left() );
			    //End script
			    if(ei > fbest) {
				   fbest = ei ;
				   bestbbInt = Xi[i] ;
				}
				//printf("fbest = %f , Ei = %f\n", fbest, ei);
				//cout << "Current Best Interval = " ;
				//for(int k=0; k<dimension; k++) cout << " " << Xi[i][k] ;
				//cout << endl ;
			    for (int j=0; j< dimension; j++) {
				   TempQue.push_back(Xi[i][j]) ;
				   TempQue_priority.push_back(ei) ;
				}
				// TempQue_priority.push_back(ei) ;
		   }
	   }
  }
}

