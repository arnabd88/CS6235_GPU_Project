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
#define intervalEpsilon 0.01
#define outputEpsilon 0.01
#define K 80 // Sampling number
#define DIM 6 // Dimension
#define GPU_CALL_INTERVAL 0.00001 
#define cpu_gpu_tolerance 0.1

#define NEW_INTV_THRESHOLD 12
#define CPU_THRESHOLD 8
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


gaol::interval cpu_Inclusion_Func_expr ( gaol::interval x, gaol::interval y, gaol::interval z, gaol::interval a, gaol::interval b, gaol::interval c ) {
   gaol::interval func;
   func = -y * z - x * a + y * b + z * c - b * c + x * (-x + y + z - a + b + c);
   return func;
}

float cpu_Func_expr ( float x, float y, float z, float a, float b, float c ) {
   float func = -y * z - x * a + y * b + z * c - b * c + x * (-x + y + z - a + b + c);
   return func;
}

__device__ float gpu_Func_expr ( float x, float y, float z, float a, float b, float c ) {
   float func;
   func = -y * z - x * a + y * b + z * c - b * c + x * (-x + y + z - a + b + c);
   return func;
}

__device__ double gpu_func_array_expr ( double* var ) {
   double func;
   func = -var[1] * var[2] - var[0] * var[3] + var[1] * var[4] + var[2] * var[5] - var[4] * var[5] + var[0] * (-var[0] + var[1] + var[2] - var[3] + var[4] + var[5]);
   return func;
}

__global__ void gpuKernel ( KernelArray<interval_gpu<double>> gpuMainQue, KernelArray<double> gpuPriQue, int dimension )
{
   //printf("Testing ................\n");
   __shared__ interval_gpu<double> SharedIntervalList[DIM] ; // Warning thrown due to blank constructor
   __shared__ double RandSampleList[DIM] ;
   __shared__ double SharedIntDimSample[DIM][K];
   double localSampleList[DIM] ;
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
void gpuHandlerThread ( KernelArray<interval_gpu<double>> gpuMainQue, KernelArray<double> gpuPriQue, int dimension) {
	   //printf("Debug: 7: Got called...\n");
       if(K*dimension > 512) {
	      cout << "Reduce the K value" << endl ;
	   }
	   else {
	       //dim3 dimBlock(K, dimension);
		   dim3 dimBlock(dimension, K);
		   dim3 dimGrid(gpuPriQue._size/dimension);

              // Initialize timer
              // cudaEvent_t start,stop;
              // float elapsed_time;
              // cudaEventCreate(&start);
              // cudaEventCreate(&stop);
              // cudaEventRecord(start,0);
		   gpuKernel<<<dimGrid,dimBlock>>>(gpuMainQue, gpuPriQue, dimension) ;
		   cudaDeviceSynchronize();
              // cudaEventRecord(stop);
              // cudaEventSynchronize(stop);
              // cudaEventElapsedTime(&elapsed_time,start, stop);
              // printf("The operation was successful, time = %2.6f\n",elapsed_time);

		   //thrust::device_vector<float> a_temp = *gpuPriQue._array ;
		  // thrust::sort(gpuPriQue._array[0], gpuPriQue._array.end[4]) ;
	   }

	   syncFlag = 1 ;
}

   int dimension = 6;
   gaol::interval x_0(-10,10), x_1(-10,10), x_2(-10,10), x_3(-10,10), x_4(-10,10), x_5(-10,10);

int main()  {
   gaol::init();
   omp_set_num_threads(NUM_THREADS);
   double gpuCallIntervalTimer, gpuStartTimer ;
   int stopGpu = 0;
   //--- Data structure for the gpu ---
   thrust::device_vector<interval_gpu<double>> gpu_interval_list ;
   thrust::device_vector<double>               gpu_interval_priority ;
 //  thrust::device_vector<float>                fbestTag ;
   float                   fbestTag ;
   KernelArray<interval_gpu<double>>           gpuMainQue ;
   KernelArray<double>                         gpuPriQue  ;             

   //--- Data structure for the cpu ---
   float  fbest = numeric_limits<float>::min();
   vector<gaol::interval> bestbbInt(dimension) ;
   deque<gaol::interval> gpubestbbInt ;
   deque<gaol::interval> MainQue ;
   deque<gaol::interval> TempQue ;
   deque<float> MainQue_priority ;
   deque<float> TempQue_priority ;
   deque<thread> ManageThreads   ;
   vector<gaol::interval> MidPoints(dimension) ;
   int count=0;
   TempQue.push_back(x_0);
   TempQue.push_back(x_1);
   TempQue.push_back(x_2);
   TempQue.push_back(x_3);
   TempQue.push_back(x_4);
   TempQue.push_back(x_5);
   vector<gaol::interval> X(dimension);  // Intervals to be used inside the while loop
   vector<gaol::interval> X1(dimension);  // Intervals to be used inside the while loop
   vector<gaol::interval> X2(dimension);  // Intervals to be used inside the while loop
   vector<vector<gaol::interval>> Xi ;
   gaol::interval FunctionBound ;

   //---- Initialise fbestTag --------
 //  fbestTag.push_back(numeric_limits<float>::min()) ;
   fbestTag = numeric_limits<float>::min();

   //---- Get the priority of the starting interval ----

   gaol::interval PriTemp = cpu_Inclusion_Func_expr( (x_0.left() + x_0.right())/2, (x_1.left() + x_1.right())/2, (x_2.left() + x_2.right())/2, (x_3.left() + x_3.right())/2, (x_4.left() + x_4.right())/2, (x_5.left() + x_5.right())/2 );

   #pragma omp parallel for
   for(int i=0; i<dimension; i++) TempQue_priority.push_back(PriTemp.left()) ;

  int loop_Counter = 0 ;
   //-- ManageThreads commented until gpu comes alive
   double startComputeTime = omp_get_wtime() ;
   while( (int)TempQue_priority.size()> 0  || (int)MainQue_priority.size()>0 || (int)ManageThreads.size()>0) {
       // printf("Start: MainQue_priority = %d \n", (int)MainQue_priority.size());
       // printf("Start: TempQue_priority = %d \n", (int)TempQue_priority.size());
       // printf("Start: ManageThreads = %d \n", (int)ManageThreads.size());
	//   printf("Debug: 6: Reached here: Temp_Queue_Size = %lu, Main_Queue_size = %lu\n", TempQue.size(), MainQue.size());
     //  cout << " Idiot!!... Terminate while :)...." << endl ;
       if(USE_GPU==1 && syncFlag == 1 && ManageThreads.size() != 0 && count>=CPU_THRESHOLD) {   // gpu calls to be joined
	       if(ManageThreads.front().joinable()) ManageThreads.front().join() ;
		   loop_Counter++ ;
		      //cout << "GPU_QUEUE_SIZE RECEIVED= " << gpuPriQue._size << endl ;
		   thrust::stable_sort_by_key(gpu_interval_priority.begin(), gpu_interval_priority.end(), gpu_interval_list.begin()) ;
		   syncFlag = 0 ;
		   count = 0 ;
		   ManageThreads.pop_front() ;
		   //MainQue.clear() ;
		   gpubestbbInt.clear();
		   for(int i=0; i < (int)gpu_interval_list.size()/dimension; i++) {   // translate gpu return list to gaol
		       for(int j=dimension-1; j>=0; j--) {
			       interval_gpu<double> ij_gpu = gpu_interval_list[i*dimension + j] ;
				   gaol::interval ij(ij_gpu.lower(), ij_gpu.upper());
				   MainQue.push_front(ij) ;
			       MainQue_priority.push_front(gpu_interval_priority[i*dimension + j]) ;
				   if(i==(int)gpu_interval_list.size()/dimension-1)
				       gpubestbbInt.push_front(ij) ;
			   }
		   }
		  fbestTag = gpu_interval_priority[((int)gpu_interval_list.size()/dimension-1)*dimension - 1];
		  if(fbestTag - fbest > 0 && fbestTag - fbest < cpu_gpu_tolerance ||
		     fbest - fbestTag > 0 && fbest - fbestTag < cpu_gpu_tolerance) stopGpu = 1 ;
		  else  stopGpu = 0;
			 //  cout << "From GPU fbestTag = " << fbestTag << endl ;
              gpu_interval_priority.clear();
			  gpu_interval_list.clear();
		  gpuStartTimer = omp_get_wtime() ;
	  }
	  if((int)TempQue.size() != 0) {
		// cout << " TempQue-Size = " << TempQue_priority.size() << endl ;
	     for(int i=0; i<(int)TempQue.size()/dimension; i++) {             // push the TempQue to the MainQue
	        for(int j=0; j < dimension; j++) {
	           MainQue.push_back(TempQue[i*dimension+j]) ;
	   	       MainQue_priority.push_back(TempQue_priority[i*dimension+j]) ;
	        }
	     }
		 TempQue_priority.clear();
		 TempQue.clear();
	  }
	    if(ManageThreads.size()==0) gpuCallIntervalTimer = omp_get_wtime() - gpuStartTimer;
	  if(USE_GPU==1 && (int)MainQue_priority.size()/dimension - CPU_THRESHOLD > NEW_INTV_THRESHOLD && ManageThreads.size()==0 && gpuCallIntervalTimer>GPU_CALL_INTERVAL && stopGpu==0) {   // minimum number of new intervals to trigger gpu
	//	cout << "Call Interval = " << gpuCallIntervalTimer << endl ;
                 //   cout << "MainQueSize = " << (int)MainQue.size() <<  endl ;
					gpuStartTimer = 0 ;
					gpuCallIntervalTimer = 0 ;
			    for(int i=CPU_THRESHOLD; i<(int)MainQue.size()/dimension; i++) {
			       for(int j=0; j<dimension; j++) {
			          interval_gpu<double> ij(MainQue[i*dimension+j].left(), MainQue[i*dimension+j].right()) ;
			      	gpu_interval_list.push_back(ij) ;
			      	//cout << " Enter here " << gpu_interval_list.size() << endl ;
			        gpu_interval_priority.push_back(MainQue_priority[i*dimension+j]) ;
			       }
			       //gpu_interval_priority.push_back(MainQue_priority[i]) ;
		        }
		       		//--- Clear the intervals that has been provided to the gpu ---
				MainQue.erase(MainQue.begin() + (CPU_THRESHOLD)*dimension , MainQue.end()) ;
				MainQue_priority.erase(MainQue_priority.begin() + (CPU_THRESHOLD)*dimension, MainQue_priority.end()) ;
				 /* KernelArray<interval_gpu<float>>*/ gpuMainQue = convertToKernel(gpu_interval_list);
				 /* KernelArray<float>*/ gpuPriQue                = convertToKernel(gpu_interval_priority);
                    //            cout << " GPU_QUEUE_SIZE SENT = " <<  gpuPriQue._size << endl ;
		      ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread
	   }
       
	   fbest = returnMax(fbest, fbestTag);
	   X.clear(); X1.clear(); X2.clear(); Xi.clear();
	   //printf("MainQueueSize = %lu, TempQueueSize = %lu\n", MainQue_priority.size(), TempQue.size());
	   for(int i=0; i<dimension; i++) {
	       X.push_back(MainQue.front()); MainQue.pop_front(); 
	       MainQue_priority.pop_front();
	   }
	  // MainQue_priority.pop_front();
	   count++ ;

      FunctionBound = cpu_Inclusion_Func_expr( X[0], X[1], X[2], X[3], X[4], X[5] );
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
         float ei = cpu_Func_expr( Xi[i][0].width()/2 + Xi[i][0].left(), Xi[i][1].width()/2 + Xi[i][1].left(), Xi[i][2].width()/2 + Xi[i][2].left(), Xi[i][3].width()/2 + Xi[i][3].left(), Xi[i][4].width()/2 + Xi[i][4].left(), Xi[i][5].width()/2 + Xi[i][5].left() );
             if(ei > fbest) {
				   fbest = ei ;
				   bestbbInt = Xi[i] ;
				}
			    for (int j=0; j< dimension; j++) {
				   TempQue.push_back(Xi[i][j]) ;
				   TempQue_priority.push_back(ei) ;
				}
				// TempQue_priority.push_back(ei) ;
		   }
	   }
  }
            cout << "Total Compute Time = " << omp_get_wtime() - startComputeTime << endl ;
			printf("fbest = %f \n", fbest);
			cout << "Current Best Interval = " ;
			for(int k=0; k<dimension; k++) cout << " " << bestbbInt[k] ; cout <<endl ;
			cout << "last Best from GPU = " ;
			for(int k=0; k<dimension; k++) cout << " " << gpubestbbInt[k] ;
			cout << endl ;
}
