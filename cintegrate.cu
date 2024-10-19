#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <errno.h>
#include <time.h>
#include <stdio.h>
#include <cmath>

#include "ex4vel.h"

#define SP 32
#define SM 2 //run with less than 3 SM's for linear interpolation
#define STEPS_PER_SEC 10000
#define STEPS 1000000000 //1 billion steps


__device__ double table_accel(int timeidx, double *d_DefaultProfile)
{
    long unsigned int tsize = sizeof(d_DefaultProfile) / sizeof(double);

    // Check array bounds for look-up table
    if(timeidx > tsize)
    {
        // printf("timeidx=%d exceeds table size = %lu and range %d to %lu\n", timeidx, tsize, 0, tsize-1);
    }

    return d_DefaultProfile[timeidx];
}

__device__ double faccel(double time, double *d_DefaultProfile)
{
    int timeidx = (int)time;
    int timeidx_next = ((int)time)+1;
    double delta_t = time - (double)((int)time);

    return (table_accel(timeidx, d_DefaultProfile) + ( (table_accel(timeidx_next, d_DefaultProfile) - 
            table_accel(timeidx, d_DefaultProfile)) * delta_t));
}


__global__ void cuda_function(double *d_sums, double start, double end, int sp, int sm){
    int my_rank = threadIdx.x; // gets rank for block
	int my_block = blockIdx.x; // gets block id to calculate rank
	double localstart, localend, a, b, n;
	my_rank += sp*my_block; // calculate rank across all blocks
    int workers = sp*sm; //calculate total workers

    localstart = (my_rank*(end/workers));
    localend = ((my_rank*(end/workers))+(end/workers)); //calculate subranges

    a = localstart;
    b = localend;
    n = STEPS/workers;

    double h = (b - a) / n;
    double sum = 0.0;

    // printf("thread %d with %lf to %lf\n", my_rank, localstart, localend);

    for(int i=0;i<n;++i){ //riemann sum for given range
        double x = a + (i*h);
        double fx = sin(x);
        sum += fx;
    }
    d_sums[my_rank] = (h*sum); //assign rank's sum to proper array index to be copied to host program
}

__global__ void cuda_test(double *d_InterpProfile, double *d_DefaultProfile, double *d_sums, double a, double b, int sp, int sm){
	int my_rank = threadIdx.x;
	int my_block = blockIdx.x;
	int localstart, localend;
	int n = sp*sm;
	my_rank += sp*my_block;

    localstart =((my_rank*(1800/n))*STEPS_PER_SEC);
    localend=(((my_rank*(1800/n))+(1800/n))*STEPS_PER_SEC);

    // printf("thread %d active\n", my_rank);

    double time, dt, sum;
    dt = 1.0/STEPS_PER_SEC;
    for(int i=localstart;i<localend;i++){
        // time you would use in your integrator and faccel(time) is the fuction to integrate
        time = 0.0 + (dt*(double)i);
        d_InterpProfile[i] = faccel(time, d_DefaultProfile);
    }

    for(int i=localstart;i<localend;i++){
        sum += d_InterpProfile[i]; //integrate off of interpolated profile
    }
    d_sums[my_rank] = sum/STEPS_PER_SEC;
}


int main(){
    struct timespec start, stop;
    double fstart, fstop;
    clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0);
	double *h_InterpProfile; //declares arrays to be copied between host and device
	double *d_InterpProfile;
    double *d_DefaultProfile;
    double *h_sums;
    double *d_sums;
	int N = SP*SM;
    h_sums = (double*)malloc(sizeof(double)*N); //allocate space on host
    d_sums = (double*)malloc(sizeof(double)*N);
	h_InterpProfile = (double*)malloc(sizeof(double)*STEPS_PER_SEC*1800); //will not throw error if incorrect size,
	d_InterpProfile = (double*)malloc(sizeof(double)*STEPS_PER_SEC*1800); //and will simply return less than expected.

	cudaMalloc((void**)&d_InterpProfile, STEPS_PER_SEC*sizeof(double)*1800); //allocate space on device
    cudaMalloc((void**)&d_DefaultProfile, sizeof(double)*1800);
    cudaMalloc((void**)&d_sums, sizeof(double)*N);

	cudaMemcpy(d_InterpProfile,h_InterpProfile,STEPS_PER_SEC*sizeof(double)*1800,cudaMemcpyHostToDevice); //initializes data
    cudaMemcpy(d_DefaultProfile,DefaultProfile,sizeof(double)*1800,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums,h_sums,sizeof(double)*N,cudaMemcpyHostToDevice);

	dim3 grid_size(SM); //delcares cuda threads to be passed in
	dim3 block_size(SP);

	cuda_test<<<grid_size,block_size>>>(d_InterpProfile, d_DefaultProfile, d_sums, 0, 1800, SP, SM); //calls kernel
    // cuda_function<<<grid_size,block_size>>>(d_sums, 0, M_PI, SP, SM);

	cudaDeviceSynchronize(); //waits for all threads to complete

	cudaMemcpy(h_InterpProfile,d_InterpProfile,STEPS_PER_SEC*sizeof(double)*1800,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sums,d_sums,sizeof(double)*N,cudaMemcpyDeviceToHost); //copy new device data back to host

    double gsum;
    for(int i=0;i<N;i++){
        gsum += h_sums[i]; //total all sums for results
    }
    clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0);
    printf("%lf seconds\n", (fstop-fstart));
    printf("final distance is:%lf\n", gsum);

	cudaFree(d_InterpProfile); //frees all device allocated memory
    cudaFree(d_DefaultProfile);
    cudaFree(d_sums);
	free(h_InterpProfile); //frees all host memory
    free(h_sums);

	return 0;
}