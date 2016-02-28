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
__shared__ __device__ static int threadCount;

#define GPU_RAND() {0.5f}

// convert floating point rgb color to 8-bit integer


__global__ void kernel_render(float3* ray_pos, float3* ray_dir, RenderParameters* paramsp, float4* inout_pixels, unsigned int* out_pixels, LightField* lightfield, float* random_array, unsigned int* source_pixels,int *launchCount,int* receiveCount, float* offsetCount)
{
    
    RenderParameters &params = *paramsp;
    int num_passes = params.raysPerPixel;
    //printf("source_pixels[1]: %d\n", source_pixels[1]);
    
    //printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
    if (threadIdx.x == 0 && threadIdx.y == 0){
      
      threadCount = 0;
        for(int i = 0; i < blockDim.x*blockDim.y*num_passes; i++) {
          sindices[i] = 0 ;
          svalues[i] =  make_float3(0,0,0);
        }
    }
    __syncthreads();
    
    //atomicAdd(threadCount,1);
    //threadCount = threadCount+1;
    //printf("Thread Count: %d\n",threadCount);
    
    int window_width = paramsp->width, window_height = paramsp->height;
    float3 min_bound = params.min_bound, max_bound = params.max_bound;
    float3 lookup_scale = {1.0f/(max_bound.x-min_bound.x), 1.0f/(max_bound.y - min_bound.y), 1.0f/(max_bound.z-min_bound.z)};
    int data_width = params.data_width, data_height = params.data_height, data_depth = params.data_depth;

    float max_scale = max(max(float(params.data_width), float(params.data_height)), float(params.data_depth));
    
    unsigned int win_x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int win_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int r_x = win_x + win_y*params.width;
    //unsigned int ray_index = r_x*paramsp->raysPerPixel
    
    /*
    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  
    */

    // Launch Photon if the corresponding pixel on the source image is white
    if(source_pixels[r_x]==0)
    {
       //printf("k: %d, source_pixels[k]: %d\n",k,source_pixels[r_x]);
       paramsp->num_zero = paramsp->num_zero+1;
       //printf("num_zero: %d\n",paramsp->num_zero);
       return;
    }
    
   float3 pos, dir, pos_c;
   
  // pos_c = params.camera_pos+params.camera_z*9.0f;
  // dir = params.camera_z*-1.0f;
   //printf("data depth 
   pos_c = make_float3(0,0,-0.5);   // co-ordinates of the center of the dot card or BOS texture in physical space
  
  
   for(int j = 0; j<paramsp->raysPerPixel;j++)
    {
        atomicAdd(launchCount, 1);
        float3 accum = make_float3(0.0,0.0,0.0);
        float phase_shift = 0;
      

        float pos_x = float(win_x%window_width)/float(window_width) - .5f;
        //float cell_width = 1.0f/float(window_width);
        //float cell_height = 1.0f/float(window_height);
        float pos_y = float(win_y%window_height)/float(window_height) - .5f;
        
        float3 pos_i = pos_c + make_float3(1,0,0)*pos_x + make_float3(0,1,0)*pos_y;
        
	pos = pos_i;
        if(r_x == 62328)
	  printf("r_x: %d, pos : %f, %f, %f\n", r_x, pos.x, pos.y, pos.z);

	// Pinhole
	float c = 2.0; // principal distance
	float3 pinhole_pos = params.camera_pos + params.camera_z*c;
	
	// find vector that connects the source pixel to the pinhole
        float3 dir_i = normalize(pinhole_pos-pos);

        //float err = 100;
 

         dir = dir_i;
	//IntersectWithVolume(pos, dir, params.min_bound, params.max_bound);

			
        pos = pos+dir*params.stepSize;
	float3 normal = {0.f,0.0f,-1.f};
        
	int steps = 1.4f/(params.stepSize);
        float old_index = 1.0;
        size_t DELTA = 1;

        for(int i = 0; i < steps; ++i) {
            pos = pos + dir*params.stepSize/old_index;
            float3 offset = pos-min_bound;
            float3 lookupfn = offset*lookup_scale; // normalized lookup

            float3 lookup = {static_cast<float>(lookupfn.x*params.data_width), static_cast<float>(lookupfn.y*params.data_height), static_cast<float>(lookupfn.z*params.data_depth)         };
           
           /* if(pos.x <= min_bound.x || pos.y <= min_bound.y || pos.z <= min_bound.z ||
               pos.x >= max_bound.x || pos.y >= max_bound.y || pos.z >= max_bound.z )
                break;
	   */

	    //Debug
	   /* if(pos.x >= min_bound.x/2 && pos.x <= max_bound.x/2 && 
	      pos.y >= min_bound.y/2 && pos.y <=max_bound.y/2)
	      break;
	      */
	
            if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
                lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA)
                continue;
	    
	    float4 val = tex3D(tex_data, lookup.x, lookup.y, lookup.z)*params.dataScalar;
                
	        val.w += 1.0; //TODO: should build this into main.cpp?

           normal = make_float3(val.x,val.y,val.z);
            old_index = val.w;

            //#if !LINE_OF_SIGHT
            dir = dir + params.stepSize*normal;
            phase_shift += val.w - 1.0;


        }
	
	
	//printf("Initial - pos: %f,%f,%f dir: %f,%f,%f\n",pos_i.x,pos_i.y,pos_i.z,dir_i.x,dir_i.y,dir_i.z);
        
	//printf("Final - pos: %f,%f,%f dir: %f,%f,%f\n", pos.x,pos.y,pos.z,dir.x,dir.y,dir.z);
        
	// Storing the position and direction of ray number j for source pixel number r_x
	unsigned int ray_index = r_x*paramsp->raysPerPixel+j;
	ray_pos[ray_index] = pos;
	ray_dir[ray_index] = dir;
	
	// Calclating the final position of the ray with respect to the camera plane

	// Define the camera plane
        float3 p_0 = params.camera_pos; // Position of camera center
	float3 n = params.camera_z;     // normal to camera plane
	// Define the ray vector 
        float3 l_0 = pos;
	float3 l = dir;
	// solve for the intersection point
	float d = dot(p_0-l_0,n)/dot(l,n);
	float3 p = d*l+l_0;
	
	pos = p; // make the intersection point as the final ray position
        
        // find the location of the intersection point on the camera plane
        float3 corner_offset = pos-params.camera_corner;
        float3 xoffset = proj3(corner_offset, params.camera_x);
        float3 yoffset = proj3(corner_offset, params.camera_y);

        unsigned int w_x = length(xoffset)*params.width-1;
        unsigned int w_y = length(yoffset)*params.height-1;
        unsigned int win_index = w_y*params.width + w_x;

	if (w_x <= params.width && w_y <= params.height)
	{
	  atomicAdd(receiveCount, 1);
	  //atomicAdd(offsetCount,length(offset));
          out_pixels[win_index] = 255; //rgbToInt(255.0,0.0,0.0);
	}
	
    }

}


#endif // KERNEL_RENDER_H
