#include <iostream>
#include <cmath>
#include <mpi.h>
#include <time.h>

#define RANGE (M_PI)
#define tscale (286.4788975)
#define ascale (.2365890)
#define vscale (67.7777777)
#define STEPS (1000000000) //are changed to accomodate requirements.

using namespace std;

double acc_function(double x);
double vel_function(double x);
double dis_function(double x);
//manually input integral of acc for vel and integral of vel for pos


////////////////////////////////////////////////////////////////////////////////
// Computes the definite integral of a given function using Left Riemann sum. //
//                                                                            //
// @param a         The lower bound of integration.                           //
// @param b         The upper bound of integration.                           //
// @param n         The number of steps to use in the approximation.          //
//                                                                            //
// @return          The approximate value of the definite integral.           //
////////////////////////////////////////////////////////////////////////////////
double riemann_sum(double a, double b, double n) 
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int idx = 0; idx < n; idx++) 
    {
        double x = a + idx * h;
        double fx = sin(x);

        // Add the value of the function at the left endpoint of each subinterval.
        sum += fx;
    }

    return h * sum;
}


int main() 
{
    struct timespec start, stop;
    double fstart, fstop;
    clock_gettime(CLOCK_MONOTONIC, &start); fstart=(double)start.tv_sec + ((double)start.tv_nsec/1000000000.0);
    int comm_sz, my_rank;
    double local_sum=0, g_sum=0;
    double local_left, local_right;
    int local_n;
    MPI_Status status;

    double a = 0.0;
    double b = (RANGE);
    double n = STEPS;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int workers = comm_sz-1;
    int worker_rank = my_rank-1;

    if(my_rank != 0) 
    {

        local_left=(worker_rank*(RANGE/workers));
        local_right=((worker_rank*(RANGE/workers))+(RANGE/workers)); //adjust range per each process
        local_n = n/workers; // amount of divisions within range - universal for all processes
        local_sum = riemann_sum(local_left, local_right, local_n);

        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        //cout<<"This is process:"<<my_rank<<" with the partial total of "<<local_sum<<endl;
       // printf("sum=%u for %u expect=%u\n", local_sum, my_rank, (my_rank*(my_rank+1)/2));

    }
    else{
        for(int i=1;i<comm_sz;i++){
            MPI_Recv(&local_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            g_sum += local_sum;
        }
    }


    // double result = left_riemann_sum(a, b, n);
    if(my_rank == 0){
        clock_gettime(CLOCK_MONOTONIC, &stop); fstop=(double)stop.tv_sec + ((double)stop.tv_nsec/1000000000.0);
        printf("%lf seconds\n", (fstop-fstart));
    }
    cout.precision(15);
    if(my_rank == 0)
        cout << "The integral of f(x) from 0.0 to " << b << " with " << n << " steps is " << g_sum << endl;
    //MPI_Bcast(&g_sum, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    //cout<<"after finalize"<<endl;
    MPI_Finalize();
    return 0;
} //not working with updated data after every integration

double acc_function(double x)
{
    return -(sin(x/tscale)*ascale);
}

double vel_function(double x)
{
    return ((-cos(x/tscale)+1)*vscale);
}

double dis_function(double x)
{
    return (vscale*(x-(tscale*(sin(x/tscale)))));
}