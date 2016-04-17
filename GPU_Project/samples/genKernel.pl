#!/usr/bin/perl
use strict;
use warnings;

#_ Name of input file
my $filename = 'sample_input_file_1';

#_ Declare the variables
my $varCount = 0;
my $function;
my $function_mod;
my @variables;
my @intervals;
my $expression_0 = "";
my $expression_1 = "";
my $expression_2 = "";
my $expression_3 = "";
my $intervalEpsilon;
my $outputEpsilon;
my $NEW_INTV_THRESHOLD;
my $CPU_THRESHOLD;
my $USE_GPU;
my $K;
my $NUM_THREADS;

#_ Open the input file in read mode
open(READ,$filename) or die "cannot open $filename";

#_ Open the output file in write mode
open(WRITE,'>workListKernel.cu');

#_ Read each line
while(my $line = <READ>) {
    my @array = split("",$line);
    
    #_ Create an identification string
    my $idString = "$array[0]" ."" ."$array[1]" ."" ."$array[2]";

    #_ Pick up the function
    if ($idString eq "-f ") {
        if ($line =~ /\"(.*?)\"/) {
           $function = $1;
           $function_mod = $1;
        }
    }
    
    #_ Count the number of variables
    if ($idString eq "-i ") {
        $varCount = ($line =~ tr/://); 
        @intervals = $line =~ /\((.*?)\)/g;
    }
    
    #_ Collect the variables
    if ($idString eq "-i ") {
        if ($line =~ /\"(.*?)\"/) {
            @variables = $1 =~ /[a-z]/g;
        }
    }

    #_ Collect other parameters
    my $part1;
    my $part2;
    ($part1, $part2) = split("=",$line);
    if ($idString eq "-ie") {
        chomp($part2);
        $intervalEpsilon = $part2;
    } elsif ($idString eq "-oe" ) {
        chomp($part2);
        $outputEpsilon = $part2;
    } elsif ($idString eq "-in") {
        chomp($part2);
        $NEW_INTV_THRESHOLD = $part2;
    } elsif ($idString eq "-cp") {
        chomp($part2);
        $CPU_THRESHOLD = $part2;
    } elsif ($idString eq "-US") {
        chomp($part2);
        $USE_GPU = $part2;
    } elsif ($idString eq "-gp") {
        chomp($part2);
        $K = $part2;
    } elsif ($idString eq "-NU") {
        chomp($part2);
        $NUM_THREADS = $part2;
    }
}

#_ Creating Argument Strings
my $argString_1 = "";
my $argString_2 = "";
my $cnt=0;
foreach my $var (@variables) {
    my $sep1 = ($cnt == 0) ? "" : ", ";
    $argString_1 = "$argString_1" ."$sep1" ."gaol::interval $var";
    $argString_2 = "$argString_2" ."$sep1" ."float $var";
    $cnt++;
}

#_ Creating Expressions
$cnt=0;
foreach my $idx (0..scalar(@variables)-1) {
    my $sep1 = ($cnt == 0) ? "" : ", ";
    $expression_0 = "$expression_0" ."$sep1" ."x_$idx($intervals[$idx])";
    $expression_1 = "$expression_1" ."$sep1" ."(x_$idx.left() + x_$idx.right())/2";
    $expression_2 = "$expression_2" ."$sep1" ."X[$idx]";
    $expression_3 = "$expression_3" ."$sep1" ."Xi[i][$idx].width()/2 + Xi[i][$idx].left()";
    $cnt++;
}

#_ Creating Modified Function
foreach my $idx (0..scalar(@variables)-1) {
    my $find = $variables[$idx];
    my $replace = "var[$idx]";
    $function_mod =~ s/\b$find\b/$replace/ig;
}

#_ Lets dump the code
print WRITE "#include <iostream>\n";
print WRITE "#include <stdio.h>\n";
print WRITE "#include <stdlib.h>\n";
print WRITE "#include <math.h>\n";
print WRITE "#include <gaol/gaol.h>\n";
print WRITE "#include \"helper_cuda.h\"\n";
print WRITE "#include \"cuda_interval_lib.h\"\n";
print WRITE "#include <limits>\n";
print WRITE "#include <vector>\n";
print WRITE "#include <queue>\n";
print WRITE "#include <deque>\n";
print WRITE "#include <thrust/device_vector.h>\n";
print WRITE "#include <thrust/host_vector.h>\n";
print WRITE "#include <thrust/sort.h>\n";
print WRITE "#include \"omp.h\"\n";
print WRITE "#include <thread>\n";
print WRITE "#define NUM_THREADS $NUM_THREADS\n";
print WRITE "#define intervalEpsilon $intervalEpsilon\n"; 
print WRITE "#define outputEpsilon $outputEpsilon\n"; 
print WRITE "#define K $K // Sampling number\n";
print WRITE "#define DIM $varCount // Dimension\n";
print WRITE "\n";
print WRITE "#define NEW_INTV_THRESHOLD $NEW_INTV_THRESHOLD\n";
print WRITE "#define CPU_THRESHOLD $CPU_THRESHOLD\n";
print WRITE "#define USE_GPU $USE_GPU\n";
print WRITE "\n";
print WRITE "using namespace std ;\n";
print WRITE "\n";
print WRITE "int syncFlag=0 ;\n";
print WRITE "\n";
print WRITE "__host__  __device__  float returnMax( float a, float b) { if(a > b) return a ; else   return b ; }\n";
print WRITE "__host__  __device__  float returnMin( float a, float b) { if(a < b) return a ; else   return b ; }\n";
print WRITE "       //--- Returns the dimension which has the largest width\n";
print WRITE "int IntervalWidth( vector<gaol::interval> X ) {\n";
print WRITE "     if(X.size()==0) { printf(\"Error: Interval list empty!!!\" ) ; exit(-1) ; }\n";
print WRITE "     float Width = X[0].width(); int   index = 0 ;\n";
print WRITE " 	 for(int i=0; i<X.size(); i++) { if(X[i].width() > Width) { Width = X[i].width() ; index = i ; } }\n";
print WRITE " 	 return index ;\n";
print WRITE "}\n";
print WRITE "\n";
print WRITE "template <typename T> struct KernelArray { T* _array ; int _size ; } ;\n";
print WRITE "template <typename T> KernelArray<T> convertToKernel(thrust::device_vector<T>& dvec)\n";
print WRITE "{ KernelArray<T> kArray ;\n";
print WRITE "   kArray._array = thrust::raw_pointer_cast(&dvec[0]);\n";
print WRITE "   kArray._size  = (int) dvec.size();\n";
print WRITE "   return kArray ; } ;\n";
print WRITE "\n";
print WRITE "\n";
#print WRITE "\n//-----------------------------------------------\n";
print WRITE "gaol::interval cpu_Inclusion_Func_expr ( $argString_1 ) {\n";
print WRITE "   gaol::interval func;\n";
print WRITE "   func = $function;\n";
print WRITE "   return func;\n";
print WRITE "}\n";
print WRITE "\n";
#print WRITE "\n//-----------------------------------------------\n";
print WRITE "float cpu_Func_expr ( $argString_2 ) {\n";
print WRITE "   float func = $function;\n";
print WRITE "   return func;\n";
print WRITE "}\n";
print WRITE "\n";
#print WRITE "\n//-----------------------------------------------\n";
print WRITE "__device__ float gpu_Func_expr ( $argString_2 ) {\n";
print WRITE "   float func;\n";
print WRITE "   func = $function;\n";
print WRITE "   return func;\n";
print WRITE "}\n";
print WRITE "\n";
#print WRITE "\n//-----------------------------------------------\n";
print WRITE "__device__ float gpu_func_array_expr ( float* var ) {\n";
print WRITE "   float func;\n";
print WRITE "   func = $function_mod;\n";
print WRITE "   return func;\n";
print WRITE "}\n";
print WRITE "\n";
print WRITE "__global__ void gpuKernel ( KernelArray<interval_gpu<double>> gpuMainQue, KernelArray<double> gpuPriQue, int dimension )\n";
print WRITE "{\n";
print WRITE "   //printf(\"Testing ................\\n\");\n";
print WRITE "   __shared__ interval_gpu<double> SharedIntervalList[DIM] ; // Warning thrown due to blank constructor\n";
print WRITE "   __shared__ double RandSampleList[DIM] ;\n";
print WRITE "   __shared__ double SharedIntDimSample[DIM][K];\n";
print WRITE "   float localSampleList[DIM] ;\n";
print WRITE "   int tix = threadIdx.x ; int tiy = threadIdx.y ;\n";
print WRITE "   float chunkSize = 0.0 ;\n";
print WRITE " //  __syncthreads();\n";
print WRITE "\n";
print WRITE "   if(tiy==0) {\n";
print WRITE "     SharedIntervalList[tix] = gpuMainQue._array[blockIdx.x*DIM + tix] ;\n";
print WRITE " //    printf(\"Copied Interval = %f from thread= %d\\n\", SharedIntervalList[tix].lower(), tix);\n";
print WRITE "     RandSampleList[tix] = (SharedIntervalList[tix].lower() + SharedIntervalList[tix].upper())/2 ;\n";
print WRITE "   }\n";
print WRITE "   __syncthreads();\n";
print WRITE "\n";
print WRITE "   chunkSize = (SharedIntervalList[tix].upper() - SharedIntervalList[tix].lower())/K ;\n";
print WRITE "   SharedIntDimSample[tix][tiy] = SharedIntervalList[tix].lower() +tiy*chunkSize + chunkSize/2 ; //-- Midpoint\n";
print WRITE "\n";
print WRITE "   __syncthreads() ;\n";
print WRITE "   for(int m=0; m<DIM; m++) localSampleList[m] = RandSampleList[m] ;\n";
print WRITE "\n";
print WRITE "  // localSampleList = RandSampleList;\n";
print WRITE "   localSampleList[tix] = SharedIntDimSample[tix][tiy] ;  // update the random sample for that thread\n";
print WRITE "\n";
print WRITE "   SharedIntDimSample[tix][tiy] = gpu_func_array_expr(localSampleList) ;\n";
print WRITE "   __syncthreads();\n";
print WRITE "\n";
print WRITE "  //----- Max reduce  for the values per tix and load in RandSampleList(reuse the allocated memory- size of dimension)\n";
print WRITE "  //----- Do a Max reduce ---\n";
print WRITE "  int size = K ;\n";
print WRITE "  for(int i=ceil((float)K/2) ; i>1; i = ceil((float)i/2)) {\n";
print WRITE "       if(tiy < i && tiy+i<size-1)\n";
print WRITE "	      SharedIntDimSample[tix][tiy] = returnMax(SharedIntDimSample[tix][tiy] ,\n";
print WRITE "		                                           SharedIntDimSample[tix][tiy+i]) ;\n";
print WRITE "	   size = i ;\n";
print WRITE "	      // SharedIntDimSample[tix][tiy] += SharedIntDimSample[tix][tiy + i] ;\n";
print WRITE "  __syncthreads() ;\n";
print WRITE "  }\n";
print WRITE "  if(K > 1 && tiy==0) SharedIntDimSample[tix][0] = returnMax(SharedIntDimSample[tix][0],\n";
print WRITE "                                                   SharedIntDimSample[tix][1]) ;\n";
print WRITE "  __syncthreads() ;\n";
print WRITE "\n";
print WRITE "  //----- Max vaue across the dimensions ----\n";
print WRITE "  size = 0;\n";
print WRITE "  for(int i=ceil((float)DIM/2) ; i > 1; i = ceil((float)i/2)) {\n";
print WRITE "      if(tix < i && tix+i<size-1)\n";
print WRITE "	     SharedIntDimSample[tix][0] = returnMax(SharedIntDimSample[tix][0],\n";
print WRITE "		                                        SharedIntDimSample[tix+i][0]) ;\n";
print WRITE "	  size = i ;\n";
print WRITE "	  __syncthreads() ;\n";
print WRITE "  }\n";
print WRITE "  //if(tix==0 && tiy==0) {\n";
print WRITE "  //   if(DIM > 1)  SharedIntDimSample[0][0] = returnMax(SharedIntDimSample[0][0],\n";
print WRITE "  //                                               SharedIntDimSample[1][0]) ;\n";
print WRITE "	 //--copy this priority value to the global memory\n";
print WRITE "  //}\n";
print WRITE "  if(tiy==0)\n";
print WRITE "     if(DIM > 1)gpuPriQue._array[blockIdx.x*DIM + tix] = returnMax(SharedIntDimSample[0][0], \n";
print WRITE "	                                                               SharedIntDimSample[1][0]);\n";
print WRITE "	 else\n";
print WRITE "	     gpuPriQue._array[blockIdx.x*DIM + tix] = SharedIntDimSample[0][0] ;\n";
print WRITE "//	 gpuPriQue._array[blockIdx.x*DIM + tix] = SharedIntDimSample[0][0] ;\n";
print WRITE "  \n";
print WRITE "\n";
print WRITE "  //----- The RandSampleList becomes the priority label array of SharedIntervalList -----\n";
print WRITE "  //----- Sort these and terminate\n";
print WRITE "\n";
print WRITE "   //---- Sorting done with a thrust call at the host(execution will still be on device)\n";
print WRITE "\n";
print WRITE "\n";
print WRITE "\n";
print WRITE "\n";
print WRITE "   __syncthreads();\n";
print WRITE "}\n";
print WRITE "\n";
print WRITE "//ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread\n";
print WRITE "void gpuHandlerThread ( KernelArray<interval_gpu<double>> gpuMainQue, KernelArray<double> gpuPriQue, int dimension) {\n";
print WRITE "	   //printf(\"Debug: 7: Got called...\\n\");\n";
print WRITE "       if(K*dimension > 512) {\n";
print WRITE "	      cout << \"Reduce the K value\" << endl ;\n";
print WRITE "	   }\n";
print WRITE "	   else {\n";
print WRITE "	       //dim3 dimBlock(K, dimension);\n";
print WRITE "		   dim3 dimBlock(dimension, K);\n";
print WRITE "		   dim3 dimGrid(gpuPriQue._size);\n";
print WRITE "		   gpuKernel<<<dimGrid,dimBlock>>>(gpuMainQue, gpuPriQue, dimension) ;\n";
print WRITE "		   cudaDeviceSynchronize();\n";
print WRITE "		   //thrust::device_vector<float> a_temp = *gpuPriQue._array ;\n";
print WRITE "		  // thrust::sort(gpuPriQue._array[0], gpuPriQue._array.end[4]) ;\n";
print WRITE "	   }\n";
print WRITE "\n";
print WRITE "	   syncFlag = 1 ;\n";
print WRITE "}\n";
print WRITE "\n";
#print WRITE "\n//-----------------------------------------------\n";
print WRITE "   int dimension = $varCount;\n";
print WRITE "   gaol::interval $expression_0;\n";
print WRITE "\n";
print WRITE "int main()  {\n";
print WRITE "   gaol::init();\n";
print WRITE "   omp_set_num_threads(NUM_THREADS);\n";
print WRITE "   //--- Data structure for the gpu ---\n";
print WRITE "   thrust::device_vector<interval_gpu<double>> gpu_interval_list ;\n";
print WRITE "   thrust::device_vector<double>               gpu_interval_priority ;\n";
print WRITE "   thrust::device_vector<float>                fbestTag ;\n";
print WRITE "   KernelArray<interval_gpu<double>>           gpuMainQue ;\n";
print WRITE "   KernelArray<double>                         gpuPriQue  ;             \n";
print WRITE "\n";
print WRITE "   //--- Data structure for the cpu ---\n";
print WRITE "   float  fbest = numeric_limits<float>::min();\n";
print WRITE "   vector<gaol::interval> bestbbInt(dimension) ;\n";
print WRITE "   deque<gaol::interval> MainQue ;\n";
print WRITE "   deque<gaol::interval> TempQue ;\n";
print WRITE "   deque<float> MainQue_priority ;\n";
print WRITE "   deque<float> TempQue_priority ;\n";
print WRITE "   deque<thread> ManageThreads   ;\n";
print WRITE "   vector<gaol::interval> MidPoints(dimension) ;\n";
print WRITE "   int addedIntervalSize = 0;               // Holds information for threshold of gpu call\n";
print WRITE "   int count=0;\n";
print WRITE "   int gpuSize=0;\n";
#print WRITE "\n//-----------------------------------------------\n";
foreach my $idx (0..scalar(@variables)-1) {
    print WRITE "   TempQue.push_back(x_$idx);\n";
}
print WRITE "   vector<gaol::interval> X(dimension);  // Intervals to be used inside the while loop\n";
print WRITE "   vector<gaol::interval> X1(dimension);  // Intervals to be used inside the while loop\n";
print WRITE "   vector<gaol::interval> X2(dimension);  // Intervals to be used inside the while loop\n";
print WRITE "   vector<vector<gaol::interval>> Xi ;\n";
print WRITE "   gaol::interval FunctionBound ;\n";
print WRITE "\n";
print WRITE "   //---- Initialise fbestTag --------\n";
print WRITE "   fbestTag.push_back(numeric_limits<float>::min()) ;\n";
print WRITE "\n";
print WRITE "   //---- Get the priority of the starting interval ----\n";
print WRITE "\n";
print WRITE "   gaol::interval PriTemp = cpu_Inclusion_Func_expr( $expression_1 );\n";
print WRITE "   for(int i=0; i<dimension; i++) TempQue_priority.push_back(PriTemp.left()) ;\n";
print WRITE "\n";
print WRITE "  int loop_Counter = 0 ;\n";
print WRITE "   //-- ManageThreads commented until gpu comes alive\n";
print WRITE "   while( (int)TempQue_priority.size()> 0  || (int)MainQue_priority.size()>0 || (int)ManageThreads.size()>0) {\n";
print WRITE "       // printf(\"Start: MainQue_priority = %d \\n\", (int)MainQue_priority.size());\n";
print WRITE "       // printf(\"Start: TempQue_priority = %d \\n\", (int)TempQue_priority.size());\n";
print WRITE "       // printf(\"Start: ManageThreads = %d \\n\", (int)ManageThreads.size());\n";
print WRITE "	//   printf(\"Debug: 6: Reached here: Temp_Queue_Size = %lu, Main_Queue_size = %lu\\n\", TempQue.size(), MainQue.size());\n";
print WRITE "     //  cout << \" Idiot!!... Terminate while :)....\" << endl ;\n";
print WRITE "       if(USE_GPU==1 && syncFlag == 1 && ManageThreads.size() != 0 && count>=CPU_THRESHOLD) {   // gpu calls to be joined\n";
print WRITE "	       if(ManageThreads.front().joinable()) ManageThreads.front().join() ;\n";
print WRITE "		   loop_Counter++ ;\n";
print WRITE "		 //  if(loop_Counter==1) {\n";
print WRITE "		 //cout << \"After the re-written priority from gpu \" << endl ;\n";
print WRITE "		 //for(int i=0; i<(int)gpu_interval_priority.size(); i++) {\n";
print WRITE "		   //interval_gpu<float> temp = gpu_interval_list[i] ;\n";
print WRITE "		   //cout << temp.lower() << \" : \" << temp.upper() << \" -- \" << gpu_interval_priority[i] << endl  ;\n";
print WRITE "		   //}\n";
print WRITE "		      //cout << \"GPU_QUEUE_SIZE = \" << (int)gpu_interval_priority.size() << endl ;\n";
print WRITE "		      cout << \"GPU_QUEUE_SIZE RECEIVED= \" << gpuPriQue._size << endl ;\n";
print WRITE "		  // }\n";
print WRITE "		   thrust::stable_sort_by_key(gpu_interval_priority.begin(), gpu_interval_priority.end(), gpu_interval_list.begin()) ;\n";
print WRITE "		 //cout << \"After Sorting \" << endl ;\n";
print WRITE "		 //for(int i=0; i<(int)gpu_interval_priority.size(); i++) {\n";
print WRITE "		 //  interval_gpu<float> temp = gpu_interval_list[i] ;\n";
print WRITE "		 //  cout << temp.lower() << \" : \" << temp.upper() << \" -- \" << gpu_interval_priority[i] << endl  ;\n";
print WRITE "		 // }\n";
print WRITE "		   syncFlag = 0 ;\n";
print WRITE "		   count = 0 ;\n";
print WRITE "		   ManageThreads.pop_front() ;\n";
print WRITE "		   //MainQue.clear() ;\n";
print WRITE "		   for(int i=0; i < (int)gpu_interval_list.size()/dimension; i++) {   // translate gpu return list to gaol\n";
print WRITE "		       for(int j=dimension-1; j>=0; j--) {\n";
print WRITE "			       interval_gpu<double> ij_gpu = gpu_interval_list[i*dimension + j] ;\n";
print WRITE "				   gaol::interval ij(ij_gpu.lower(), ij_gpu.upper());\n";
print WRITE "				   MainQue.push_front(ij) ;\n";
print WRITE "			       MainQue_priority.push_front(gpu_interval_priority[i*dimension + j]) ;\n";
print WRITE "			   }\n";
print WRITE "			   //MainQue_priority.push_front(gpu_interval_priority[i]) ;\n";
print WRITE "		   }\n";
print WRITE "              	  gpu_interval_priority.clear();\n";
print WRITE "			  gpu_interval_list.clear();\n";        
print WRITE "	  }\n";
print WRITE "	  if((int)TempQue.size() != 0) {\n";
print WRITE "		// cout << \" TempQue-Size = \" << TempQue_priority.size() << endl ;\n";
print WRITE "	     for(int i=0; i<(int)TempQue.size()/dimension; i++) {             // push the TempQue to the MainQue\n";
print WRITE "	        for(int j=0; j < dimension; j++) {\n";
print WRITE "	           MainQue.push_back(TempQue[i*dimension+j]) ;\n";
print WRITE "	   	       MainQue_priority.push_back(TempQue_priority[i*dimension+j]) ;\n";
print WRITE "	        }\n";
print WRITE "	   	 //MainQue_priority.push_back(TempQue_priority[i]) ;\n";
print WRITE "         //printf(\"Update-2: MainQue_priority = %d \\n\", (int)MainQue_priority.size());\n";
print WRITE "	     }\n";
print WRITE "		 TempQue_priority.clear();\n";
print WRITE "		 TempQue.clear();\n";
print WRITE "	  }\n";
print WRITE "	  if(USE_GPU==1 && (int)MainQue_priority.size()/dimension - CPU_THRESHOLD > NEW_INTV_THRESHOLD && ManageThreads.size()==0 ) {   // minimum number of new intervals to trigger gpu\n";
print WRITE "			  //gpu_interval_list.clear();\n";
print WRITE "			  //gpu_interval_priority.clear();\n";
print WRITE "                    cout << \"MainQueSize = \" << (int)MainQue.size() <<  endl ;\n";
print WRITE "			    for(int i=CPU_THRESHOLD; i<(int)MainQue.size()/dimension; i++) {\n";
print WRITE "			       for(int j=0; j<dimension; j++) {\n";
print WRITE "			          interval_gpu<double> ij(MainQue[i*dimension+j].left(), MainQue[i*dimension+j].right()) ;\n";
print WRITE "			      	gpu_interval_list.push_back(ij) ;\n";
print WRITE "			      	//cout << \" Enter here \" << gpu_interval_list.size() << endl ;\n";
print WRITE "			        gpu_interval_priority.push_back(MainQue_priority[i*dimension+j]) ;\n";
print WRITE "			       }\n";
print WRITE "			       //gpu_interval_priority.push_back(MainQue_priority[i]) ;\n";
print WRITE "		        }\n";
print WRITE "		        //cout << \"Starting  New GPU call of size \" << gpu_interval_priority.size() << endl ;\n";
print WRITE "		        //for(int i=0; i<(int)gpu_interval_priority.size(); i++) {\n";
print WRITE "			//	  interval_gpu<double> temp = gpu_interval_list[i] ;\n";
print WRITE "		        //  cout << temp.lower() << \" : \" << temp.upper() << \" -- \" << gpu_interval_priority[i] << endl  ;\n";
print WRITE "			//	}\n";
print WRITE "		       		//--- Clear the intervals that has been provided to the gpu ---\n";
print WRITE "				MainQue.erase(MainQue.begin() + (CPU_THRESHOLD)*dimension , MainQue.end()) ;\n";
print WRITE "				MainQue_priority.erase(MainQue_priority.begin() + (CPU_THRESHOLD)*dimension, MainQue_priority.end()) ;\n";
print WRITE "				 /* KernelArray<interval_gpu<float>>*/ gpuMainQue = convertToKernel(gpu_interval_list);\n";
print WRITE "				 /* KernelArray<float>*/ gpuPriQue                = convertToKernel(gpu_interval_priority);\n";
print WRITE "                                cout << \" GPU_QUEUE_SIZE SENT = \" <<  gpuPriQue._size << endl ;\n";
print WRITE "		      ManageThreads.push_back(thread(gpuHandlerThread, gpuMainQue, gpuPriQue, dimension)) ; // Trigger the gpu thread\n";
print WRITE "	   }\n";
print WRITE "       \n";
print WRITE "	   fbest = returnMax(fbest, fbestTag.front());\n";
print WRITE "	   X.clear(); X1.clear(); X2.clear(); Xi.clear();\n";
print WRITE "	   //printf(\"MainQueueSize = %lu, TempQueueSize = %lu\\n\", MainQue_priority.size(), TempQue.size());\n";
print WRITE "	   for(int i=0; i<dimension; i++) {\n";
print WRITE "	       X.push_back(MainQue.front()); MainQue.pop_front(); \n";
print WRITE "	       MainQue_priority.pop_front();\n";
print WRITE "	   }\n";
print WRITE "	  // MainQue_priority.pop_front();\n";
print WRITE "	   count++ ;\n";
print WRITE "\n";
print WRITE "      FunctionBound = cpu_Inclusion_Func_expr( $expression_2 );\n";
print WRITE "           if ( FunctionBound.right() < fbest  ||  X[IntervalWidth(X)].width() <= intervalEpsilon || FunctionBound.width() <= outputEpsilon ) {\n";
print WRITE "	        //printf(\"GetNextElement\\n\");\n";
print WRITE "			//cout << \"Current-Size = \" << (int)MainQue_priority.size()/dimension << endl ;\n";
print WRITE "	   }\n";
print WRITE "	   else {\n";
print WRITE "	       for(int i=0; i<dimension; i++) {\n";
print WRITE "		       if(i == IntervalWidth(X)) {\n";
print WRITE "		              gaol::interval a(X[i].left(), X[i].left() + X[i].width()/2 ) ;\n";
print WRITE "		              gaol::interval b(X[i].left() + X[i].width()/2, X[i].right() ) ;\n";
print WRITE "			          X1.push_back(a); \n";
print WRITE "			          X2.push_back(b);\n";
print WRITE "			   } else {\n";
print WRITE "			         X1.push_back(X[i]); X2.push_back(X[i]) ;\n";
print WRITE "			   }\n";
print WRITE "		   }\n";
print WRITE "		   Xi.push_back(X1); Xi.push_back(X2) ;\n";
print WRITE "		   for(int i=0; i< 2; i++) {\n";
print WRITE "         float ei = cpu_Func_expr( $expression_3 );\n";
print WRITE "             if(ei > fbest) {\n";
print WRITE "				   fbest = ei ;\n";
print WRITE "				   bestbbInt = Xi[i] ;\n";
print WRITE "				}\n";
print WRITE "				//printf(\"fbest = %f , Ei = %f\\n\", fbest, ei);\n";
print WRITE "				//cout << \"Current Best Interval = \" ;\n";
print WRITE "				//for(int k=0; k<dimension; k++) cout << \" \" << Xi[i][k] ;\n";
print WRITE "				//cout << endl ;\n";
print WRITE "			    for (int j=0; j< dimension; j++) {\n";
print WRITE "				   TempQue.push_back(Xi[i][j]) ;\n";
print WRITE "				   TempQue_priority.push_back(ei) ;\n";
print WRITE "				}\n";
print WRITE "				// TempQue_priority.push_back(ei) ;\n";
print WRITE "		   }\n";
print WRITE "	   }\n";
print WRITE "  }\n";
print WRITE "			printf(\"fbest = %f \\n\", fbest);\n";
print WRITE "			cout << \"Current Best Interval = \" ;\n";
print WRITE "			for(int k=0; k<dimension; k++) cout << \" \" << bestbbInt[k] ;\n";
print WRITE "			cout << endl ;\n";



print WRITE "}\n";


#_ Close the handles\n";
close(WRITE);
close(READ);

#system("/usr/local/cuda-7.5/bin/nvcc -I/usr/local/cuda-7.5/include/ -I/usr/local/cuda-7.5/samples/common/inc/ -I/usr/local/cuda-7.5/samples/6_Advanced/interval/ -g -Xcompiler -fopenmp --std=c++11 workListKernel.cu -I../../Gaol/gaol-4.2.0/gaol/.libs/ -L../../Gaol/gaol-4.2.0/gdtoa/.libs/ -I. -lgaol -lm -lultim -lgdtoa -lgomp");
#system("./a.out > run.log");

