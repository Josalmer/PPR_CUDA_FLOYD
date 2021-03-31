#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blocksize 256

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

//**************************************************************************
// FLOYD 2D BLOCKS
__global__ void floyd_kernel(int *M, const int nverts, const int k) {
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int index = i * nverts + j;
	if (i < nverts && j < nverts) {
		int Mij = M[index];
		if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			M[index] = (Mij > Mikj) ? Mikj : Mij;
		}
	}
}

//**************************************************************************
// FIND MAX IN VECTOR
__global__ void reduceMax(int * V_in, int * V_out, const int N) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : -1);
	sdata[tid] = (((i + blockDim.x) < N) && V_in[i] < V_in[i + blockDim.x] ? V_in[i + blockDim.x] : sdata[tid]);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1) {
	  if (tid < s) {
		if(sdata[tid] < sdata[tid+s]) {
                    sdata[tid] = sdata[tid+s];	
		}
	  }
	  __syncthreads();
	}
	if (tid == 0) {
		V_out[blockIdx.x] = sdata[0];
	}
}

int main(int argc, char *argv[])
{

	if (argc != 2)
	{
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return (-1);
	}

	//Get GPU information
	int devID;
	cudaDeviceProp props;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if (err != cudaSuccess)
	{
		cout << "ERRORRR" << endl;
	}

	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]); // Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2 * sizeof(int);
	int *d_In_M = NULL;

	err = cudaMalloc((void **)&d_In_M, size);
	if (err != cudaSuccess)
	{
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	// GPU phase
	double t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		cout << "ERROR COPIA A GPU" << endl;
	}

	for (int k = 0; k < niters; k++)
	{
		//printf("CUDA kernel launch \n");
		int threadsPerDim = sqrt(blocksize);
		dim3 threadsPerBlock (threadsPerDim, threadsPerDim);
		dim3 numBlocks(
			ceil((float)nverts/threadsPerBlock.x),
			ceil((float)nverts/threadsPerBlock.y)
		);

		floyd_kernel<<<numBlocks, threadsPerBlock>>>(d_In_M, nverts, k);
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch kernel! ERROR= %d\n", err);
			exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu = cpuSecond() - t1;

	cout << "Tiempo gastado GPU= " << Tgpu << endl
		 << endl;

	// CPU phase
	t1 = cpuSecond();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for (int k = 0; k < niters; k++)
	{
		kn = k * nverts;
		for (int i = 0; i < nverts; i++)
		{
			in = i * nverts;
			for (int j = 0; j < nverts; j++)
				if (i != j && i != k && j != k)
				{
					inj = in + j;
					A[inj] = min(A[in + k] + A[kn + j], A[inj]);
				}
		}
	}

	double t2 = cpuSecond() - t1;
	cout << "Tiempo gastado CPU= " << t2 << endl
		 << endl;
	cout << "Ganancia= " << t2 / Tgpu << endl;

	for (int i = 0; i < nverts; i++)
		for (int j = 0; j < nverts; j++)
			if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0)
				cout << "Error (" << i << "," << j << ")   " << c_Out_M[i * nverts + j] << "..." << G.arista(i, j) << endl;


	// c_d Minimum computation on GPU
	dim3 threadsPerBlock(blocksize);
	dim3 numBlocks( ceil ((float)(nverts2 / 2)/threadsPerBlock.x));

	// Maximum vector on CPU
	int * vmax;
	vmax = (int*) malloc(numBlocks.x*sizeof(int));

	// Maximum vector  to be computed on GPU
	int *vmax_d; 
	cudaMalloc ((void **) &vmax_d, sizeof(int)*numBlocks.x);

	int smemSize = threadsPerBlock.x*sizeof(int);

	// Kernel launch to compute Minimum Vector
	reduceMax<<<numBlocks, threadsPerBlock, smemSize>>>(c_Out_M,vmax_d, nverts2);


	/* Copy data from device memory to host memory */
	cudaMemcpy(vmax, vmax_d, numBlocks.x*sizeof(int),cudaMemcpyDeviceToHost);

	// Perform final reduction in CPU
	int max_gpu = -1;
	for (int i=0; i<numBlocks.x; i++) {
		max_gpu =max(max_gpu,vmax[i]);
	}

	cout << endl << " Camino más largo entre los caminos mínimos = " << max_gpu << endl;
}
