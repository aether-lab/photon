#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>


// http://www.noktec.be/archives/1496
//#define CUDA_SAFE_CALL(x); checkCudaErrors(x);
#define cutilDeviceSynchronize cudaDeviceSynchronize

#define cutilCheckMsg(x) getLastCudaError(x)

/* Change Timers
   From:

    unsigned int timer=0;
    cutCreateTimer(&timer);
    cutResetTimer(&timer);
    cutStartTimer(&timer);
    ...ur code...
    cutStopTimer(&timer);
    float time = cutGetTimerValue(&timer);
    cutDeleteTimer(&timer);

  To:
    StopwatchInterface *timer=NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    ...ur code...
    sdkStopTimer(&timer);
    float time = sdkGetTimerValue(&timer);
    cutDeleteTimer(&timer);

 */
