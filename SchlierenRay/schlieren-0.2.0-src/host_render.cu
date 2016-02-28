/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2012-2013
  Scientific Computing and Imaging Institute, University of Utah

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/

#define DEBUG 0

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <teem/nrrd.h>


// includes, GL
#include "opengl_include.h"
#include <float.h>
#include <assert.h>

// includes
#include "cutil.h"
#include "cutil_math.h"
#include <cuda.h>
#include "cuda_gl_interop.h"
#include <cstdlib>

#include "RenderParameters.h"
#include "kernel_render.h"
#include "kernel_filter.h"

RenderParameters* dparams;
cudaArray* data_array = 0, *texture_array = 0, *color_array = 0;
unsigned int last_width, last_height;
unsigned int* d_out = 0;
float4* d_inout = 0;
float* d_rand_x = 0;

void Host_CopyMemory(RenderParameters* params);
void Host_Resize(RenderParameters* paramsp);

__host__ int rgbToIntHost(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__host__ float getRandom()
{
  return drand48();
}

extern "C"
{

    void Host_Init(RenderParameters* paramsp)
    {
        cudaMalloc((void**)&dparams, sizeof(RenderParameters));
        cudaMemcpy(dparams, paramsp, sizeof(RenderParameters),cudaMemcpyHostToDevice);

        //setup data texture
        tex_data.addressMode[0] = cudaAddressModeClamp;
        tex_data.addressMode[1] = cudaAddressModeClamp;
        tex_data.filterMode = cudaFilterModeLinear;
        tex_data.normalized = false;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaExtent extent = make_cudaExtent(paramsp->data_width, paramsp->data_height, paramsp->data_depth);
        checkCudaErrors( cudaMalloc3DArray(&data_array, &channelDesc, extent) );
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)paramsp->data, extent.width*sizeof(float4), extent.width, extent.height);
        copyParams.dstArray = data_array;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(  cudaMemcpy3D(&copyParams) );

        cudaBindTextureToArray(tex_data, data_array, channelDesc);

        //setup cutoff texture
        if (paramsp->cutoff  == CUTOFF_IMAGE)
        {
            tex_cutoff.addressMode[0] = cudaAddressModeClamp;
            tex_cutoff.addressMode[1] = cudaAddressModeClamp;
            tex_cutoff.filterMode = cudaFilterModeLinear;
            tex_cutoff.normalized = true;

            cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
            cudaMallocArray(&texture_array, &channelDesc2, paramsp->cutoffSize.x, paramsp->cutoffSize.y);
            cudaMemcpyToArray(texture_array, 0, 0, paramsp->cutoff_rgb, paramsp->cutoffSize.x*paramsp->cutoffSize.y*sizeof(float4), cudaMemcpyHostToDevice);

            cudaBindTextureToArray(tex_cutoff, texture_array, channelDesc2);
            paramsp->cutoff_dirty = false;
        }

        Host_Resize(paramsp);
    }

    void Host_Render(RenderParameters* paramsp)
    {
        printf("rendering...");
        if (last_width != paramsp->width || last_height != paramsp->height)
            Host_Resize(paramsp);

        RenderParameters& params = *paramsp;
        Host_CopyMemory(paramsp);

        printf("min_bound : %f, %f, %f\n", params.min_bound.x, params.min_bound.y, params.min_bound.z);
        printf("max_bound : %f, %f, %f\n", params.max_bound.x, params.max_bound.y, params.max_bound.z);

        // define block and grid by the CUDA kernel
        dim3 block(16,16,1);
        dim3 grid((params.width/block.x),(params.height/block.y),1);   //positions go 0 to 100, which maps to -1 to 1 on each lightplace axis

        // compute four random numbers to calculate initial position of photon
        paramsp->rand1 = drand48();
        paramsp->rand2 = drand48();
        paramsp->rand3 = drand48();
        paramsp->rand4 = drand48();
        
        // allocate space on the device for the results
        cudaMemcpy(dparams, paramsp, sizeof(RenderParameters),cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
        
        for(int i =0; i < paramsp->numRenderPasses; i++)
        {
            printf("i : %d",i);
            // render image
            kernel_render<<< grid, block>>>(dparams, d_inout, d_out, d_rand_x);
            cudaThreadSynchronize();
            paramsp->passes+=paramsp->raysPerPixel;
        }

        // register photons and smooth image
        kernel_filter<<< grid, block>>>(dparams, d_inout, d_out);
        
        // copy image array to CPU
        cudaMemcpy(paramsp->out_rgb, d_out, sizeof(unsigned int)*params.width*params.height, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        // display image on screen
        //glDrawPixels(params.width, params.height, GL_RGBA, GL_UNSIGNED_BYTE, paramsp->out_rgb);
        
        printf(" rendering finished\n");
    }

    void Host_Clear(RenderParameters* paramsp)
    {
        if (!d_inout)
            return;
        cudaMemcpy(d_inout, paramsp->inout_rgb, sizeof(float4)*paramsp->width*paramsp->height, cudaMemcpyHostToDevice);
    }

    void Host_Kill()
    {
        /* This functions sets all pointers to zero and frees GPU memory*/
        
        RenderParameters* dparams;
        cudaArray* data_array = 0, *texture_array = 0, *color_array = 0;
        cudaFree(d_inout);
        cudaFree(dparams);
        cudaFree(d_out);
        cudaFree(d_rand_x);

        cudaUnbindTexture (tex_data);
        checkCudaErrors (cudaFreeArray (data_array));
        cudaUnbindTexture (tex_data2);
        checkCudaErrors (cudaFreeArray (texture_array));
        cudaUnbindTexture (tex_cutoff);
        checkCudaErrors (cudaFreeArray (color_array));
    }


}

void Host_CopyMemory(RenderParameters* paramsp)
{
    //TODO: NOTE: for debugging perposes only memcopy, however need to support size changes
    if (paramsp->cutoff_dirty)
    {
        cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
        cudaMemcpyToArray(texture_array, 0, 0, paramsp->cutoff_rgb, paramsp->cutoffSize.x*paramsp->cutoffSize.y*sizeof(float4), cudaMemcpyHostToDevice);
        paramsp->cutoff_dirty = false;
    }
}

void Host_Resize(RenderParameters* paramsp)
{
    printf("resizing to %d %d \n", paramsp->width, paramsp->height);
    paramsp->passes = 0;
    int window_size = paramsp->width*paramsp->height;
    if (d_inout)
        cudaFree(d_inout);

    cudaMalloc((void**)&d_inout, sizeof(float4)*paramsp->width*paramsp->height);
    for(size_t i = 0; i < paramsp->width*paramsp->height; i++)
        paramsp->inout_rgb[i] = make_float4(0,0,0,0);
    cudaMemcpy(d_inout, paramsp->inout_rgb, sizeof(float4)*paramsp->width*paramsp->height, cudaMemcpyHostToDevice);

    if (d_out)
        cudaFree(d_out);
    cudaMalloc((void**)&d_out, sizeof(unsigned int)*paramsp->width*paramsp->height);
    last_width = paramsp->width;
    last_height = paramsp->height;

    if (paramsp->out_rgb)
        free(paramsp->out_rgb);
    paramsp->out_rgb = (unsigned int*)malloc(sizeof(unsigned int)*paramsp->width*paramsp->height);

    if (d_rand_x)
        cudaFree(d_rand_x);

    paramsp->random_array  = (float*)malloc(sizeof(float)*window_size);

    for(int i =0;i<window_size;i++)
        paramsp->random_array[i] = getRandom();

    cudaMalloc((void**)&d_rand_x, window_size*sizeof(float));
    cudaMemcpy(d_rand_x, paramsp->random_array,sizeof(float)*window_size, cudaMemcpyHostToDevice);
}

