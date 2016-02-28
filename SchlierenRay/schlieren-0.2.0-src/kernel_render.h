#ifndef KERNEL_RENDER_H
#define KERNEL_RENDER_H

texture<float4, 3> tex_data;
texture<float, 3> tex_data2;
texture<float4, 2> tex_cutoff;
texture<float4, 3> tex_color;

#include "kernel_functions.h"
#include "kernel_cutoff.h"

#define MAX_NUM_PASSES 3

__shared__ __device__ float3 svalues[256*MAX_NUM_PASSES];
__shared__ __device__ unsigned int sindices[256*MAX_NUM_PASSES];

#define GPU_RAND() {0.5f}

// convert floating point rgb color to 8-bit integer


__global__ void kernel_render(RenderParameters* paramsp, float4* inout_pixels, unsigned int* out_pixels, float* random_array)
{

    RenderParameters &params = *paramsp;
    int num_passes = params.raysPerPixel;

    if (threadIdx.x == 0 && threadIdx.y == 0){
        for(int i = 0; i < blockDim.x*blockDim.y*num_passes; i++) {
          sindices[i] = 0 ;
          svalues[i] =  make_float3(0,0,0);
        }
    }
    __syncthreads();


    int window_width = paramsp->width, window_height = paramsp->height;
    float3 min_bound = params.min_bound, max_bound = params.max_bound;

    float3 lookup_scale = {1.0f/(max_bound.x-min_bound.x), 1.0f/(max_bound.y - min_bound.y), 1.0f/(max_bound.z-min_bound.z)};
    int data_width = params.data_width, data_height = params.data_height, data_depth = params.data_depth;

	// calculate grid spacings for gradient calculation
	float grid_x = (max_bound.x - min_bound.x)/data_width;
	float grid_y = (max_bound.y - min_bound.y)/data_height;
	float grid_z = (max_bound.z - min_bound.z)/data_depth;

    int winsize = params.width*params.height;
    float3 dir = params.camera_z*-1.0f;
    float3 original_dir = normalize(dir);
    unsigned int win_x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int win_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int sindex = threadIdx.y*blockDim.x + threadIdx.x;// shared memory index
    unsigned int r_x = win_x + win_y*params.width;

    unsigned int index = r_x%(winsize);
    float3 accum = make_float3(0.0,0.0,0.0);

    float rand_t2 = paramsp->rand3;
    float rand_x2 = paramsp->rand1;
    float rand_y2 = paramsp->rand2;
    float rand_scale = 1.0;

    for(int pass = 0; pass < num_passes; pass++) 
    {
        float phase_shift = 0;
        float randx = random_array[int(r_x*rand_t2 + blockIdx.x + blockIdx.y + pass + rand_x2*winsize)%winsize]*rand_scale;//GPU_RAND();
        float randy = random_array[int(r_x + blockIdx.x + blockIdx.y + rand_t2*win_x + pass + rand_y2*winsize)%winsize]*rand_scale;//GPU_RAND();
        float randt = random_array[int(r_x + blockIdx.x + blockIdx.y + pass + rand_t2*winsize)%winsize]*rand_scale;//GPU_RAND();

        float3 pos;
        dir = original_dir;
        pos = params.camera_pos+params.camera_z*9.0f;

        float pos_x = float(win_x%window_width)/float(window_width) - .5f;
        float cell_width = 1.0f/float(window_width);
        float cell_height = 1.0f/float(window_height);
        float pos_y = float(win_y%window_height)/float(window_height) - .5f;
        float offx = cos(randx*M_PI*2.0);
        float offy = sin(randx*M_PI*2.0);
        offx = offx*randy*cell_width*0.5;
        offy = offy*randy*cell_height*0.5;
        pos = pos+params.camera_x*pos_x + params.camera_x*(randx*1.0f)*cell_width;
        pos = pos+params.camera_y*pos_y + params.camera_y*(randy*1.0f)*cell_height;

        IntersectWithVolume(pos, dir, params.min_bound, params.max_bound);

        pos = pos+dir*randt*params.stepSize;

        float3 normal = {0.f,0.0f,-1.f};
        int steps = 1.4f/(params.stepSize);
        float old_index = 1.0;
        size_t DELTA = 1;

        int i = 0;
        for(i = 0; i < steps; ++i) 
        {
            pos = pos + dir*params.stepSize/old_index;
            float3 offset = pos-min_bound;
            float3 lookupfn = offset*lookup_scale; // normalized lookup
            float3 lookup = {static_cast<float>(lookupfn.x*params.data_width), static_cast<float>(lookupfn.y*params.data_height), static_cast<float>(lookupfn.z*params.data_depth) };

            if(pos.x <= min_bound.x || pos.y <= min_bound.y || pos.z <= min_bound.z ||
               pos.x >= max_bound.x || pos.y >= max_bound.y || pos.z >= max_bound.z )
                break;

            if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
                lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA)
                continue;

            float4 val = tex3D(tex_data, lookup.x, lookup.y, lookup.z)*params.dataScalar;
            val.w += 1.0; //TODO: should build this into main.cpp?

            // calculate gradient
            normal = make_float3(val.x/(2*grid_x),val.y/(2*grid_y),val.z/(2*grid_z));
            //normal = GRADIENT_GET(lookupf.x, lookupf.y, lookupf.z);

            old_index = val.w;
            dir = dir + params.stepSize*normal;
            phase_shift += val.w - 1.0;
        }

        if (dot(params.camera_z*-1.0f, normalize(dir)) < 0.0)
            continue;

        //#if PROJECT_TO_FILM
        if (params.projectionDistance > 0.0)
        {
            //project to film plane
            float3 film_pos = params.camera_pos;
            IntersectWithVolume(film_pos, params.camera_z, params.min_bound, params.max_bound);
            film_pos = film_pos+params.camera_z*-params.projectionDistance;
            IntersectWithPlane(pos, dir, params.camera_z, film_pos);
        }
        // #endif
        phase_shift *= 2.0*M_PI/(500e-9)*paramsp->stepSize;

        float3 offset = normalize(dir)-original_dir;
        float3 corner_offset = pos-params.camera_corner;
        float signedx =   dot(offset,params.camera_x);
        float signedy =   dot(offset,params.camera_y);
        float3 xoffset = proj3(corner_offset, params.camera_x);
        float3 yoffset = proj3(corner_offset, params.camera_y);

        unsigned int w_x = length(xoffset)*params.width;
        unsigned int w_y = length(yoffset)*params.height;
        unsigned int win_index = w_y*params.width + w_x;
        if (w_x >= params.width || w_y >= params.height)
            continue;

        float2 cutoff_offset = make_float2(signedx, signedy);
        float3 c;
        kernel_cutoff(params, cutoff_offset, phase_shift, c);
        accum = accum*params.stepSize;

        sindices[sindex] =win_index;
        svalues[sindex] = make_float3(c.x,c.y,c.z);
        sindex += 256;
    }

   __syncthreads();
   if (threadIdx.x == 0 && threadIdx.y == 0)
   {
       unsigned int num = blockDim.x*blockDim.y*num_passes;
       for(int i = 0; i < num; i++) 
       {
           float3 c = svalues[i];
           unsigned int ind = sindices[i];
           if (ind <  winsize) 
             inout_pixels[ind] += make_float4(c.x,c.y,c.z,1.0);
       }
   }

}


#endif // KERNEL_RENDER_H
