# PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES AY 23-24
<h3>AIM: To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.</h3>
<h3> NAME:SREE NIVEDITAA SARAVANAN</h3>
<h3>REGISTER NO:212223230213</h3>
<h3>EX. NO:3</h3>
<h3>DATE:02-04-2024</h3>
<h1> <align=center> PARALLEL REDUCTION USING UNROLLING TECHNIQUES </h3>
  Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.</h3>

## AIM:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Initialization and Memory Allocation
2.	Define the input size n.
3.	Allocate host memory (h_idata and h_odata) for input and output data.
Input Data Initialization
4.	Initialize the input data on the host (h_idata) by assigning a value of 1 to each element.
Device Memory Allocation
5.	Allocate device memory (d_idata and d_odata) for input and output data on the GPU.
Data Transfer: Host to Device
6.	Copy the input data from the host (h_idata) to the device (d_idata) using cudaMemcpy.
Grid and Block Configuration
7.	Define the grid and block dimensions for the kernel launch:
8.	Each block consists of 256 threads.
9.	Calculate the grid size based on the input size n and block size.
10.	Start CPU Timer
11.	Initialize a CPU timer to measure the CPU execution time.
12.	Compute CPU Sum
13.	Calculate the sum of the input data on the CPU using a for loop and store the result in sum_cpu.
14.	Stop CPU Timer
15.	Record the elapsed CPU time.
16.	Start GPU Timer
17.	Initialize a GPU timer to measure the GPU execution time.
Kernel Execution
18.	Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.
Data Transfer: Device to Host
19.	Copy the result data from the device (d_odata) to the host (h_odata) using cudaMemcpy.
20.	Compute GPU Sum
21.	Calculate the final sum on the GPU by summing the elements in h_odata and store the result in sum_gpu.
22.	Stop GPU Timer
23.	Record the elapsed GPU time.
24.	Print Results
25.	Display the computed CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.
Memory Deallocation
26.	Free the allocated host and device memory using free and cudaFree.
27.	Exit
28.	Return from the main function.

## PROGRAM:
```
%%cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce(int *g_in, int *g_out, int n) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_in[i] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(int);
    
    int *h_in = (int*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = 1;  // All 1s for easy verification
    
    int block = 256;
    int grid = (n + block - 1) / block;
    int *h_out = (int*)malloc(grid * sizeof(int));
    
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, grid * sizeof(int));
    
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    reduce<<<grid, block, block * sizeof(int)>>>(d_in, d_out, n);
    
    cudaMemcpy(h_out, d_out, grid * sizeof(int), cudaMemcpyDeviceToHost);
    
    int sum = 0;
    for (int i = 0; i < grid; i++) sum += h_out[i];
    
    printf("Sum: %d (expected %d)\n", sum, n);
    printf("%s\n", (sum == n) ? "Success!" : "Failed!");
    
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    
    return 0;
}
```

## OUTPUT:

![Screenshot 2024-04-05 132641](https://github.com/2005Mukesh/PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES-AY-23-24/assets/138849308/227f9011-3214-4f40-92f7-a98c1f34a35b)

## RESULT:
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that  1048576 has executed with less elapsed time than 1048576 with blocks 2.73 ms,116.58 ms.
