#ifndef KERNEL_RENDER_H
#define KERNEL_RENDER_H

texture<float4, 3> tex_data;
texture<float, 3> tex_data2;
texture<float4, 2> tex_cutoff;
texture<float4, 3> tex_color;

#include "kernel_functions.h"
#include "kernel_cutoff.h"

#define MAX_NUM_PASSES 1

__shared__ __device__ float3 svalues[1024*MAX_NUM_PASSES];
__shared__ __device__ unsigned int sindices[1024*MAX_NUM_PASSES];
__shared__ __device__ static int threadCount;
//__shared__ __device__ float random_number;
#define GPU_RAND() {0.5f}

// convert floating point rgb color to 8-bit integer

__device__ float3 rayPlaneIntersection(float3 p_0, float3 n, float3 l_0, float3 l) 
{
	//******************************* Find location of intersection of the ray with a plane
	// Plane : p_0 - point on the plane, n - normal to the plane
	// Ray : l_0 - origin, l - direction
	// Intersection : d - angle between ray and normal vector, p - position on the plane 
	float3 p;
	float d1 = dot(l,n);
	float d = dot(p_0-l_0,n)/d1;
	p = d*l + l_0;

	return p;
}

__device__ float random_single(unsigned int seed)
{

  //unsigned int seed = time(0);
  //unsigned int seed = threadIdx.x;
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* the seed can be the same for each core, here we pass the time in from the CPU */
  /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
  /* the offset is how much extra we advance in the sequence for each call, can be 0 */

  /* we have to initialize the state */
  curand_init(seed, blockIdx.x, 0, &state);

  float rand_num = curand_uniform(&state);
  return rand_num;
}
 
__global__ void kernel_render(float3* ray_pos, float3* ray_dir, RenderParameters* paramsp, float4* inout_pixels, unsigned int* out_pixels, int *launchCount,int* receiveCount, int* threshCount)
{

	RenderParameters &params = *paramsp; // make this array in local memory to ensure speed up.
	//int num_passes = params.raysPerPixel;


	//int window_width = params.width, window_height = params.height;
	float3 min_bound = params.min_bound, max_bound = params.max_bound;
	float3 lookup_scale = {1.0f/(max_bound.x-min_bound.x), 1.0f/(max_bound.y - min_bound.y), 1.0f/(max_bound.z-min_bound.z)};
	int data_width = params.data_width, data_height = params.data_height, data_depth = params.data_depth;

	float max_scale = max(max(float(params.data_width), float(params.data_height)), float(params.data_depth));

	/*	
	unsigned int win_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int win_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int r_x = win_x + win_y*params.width;

	unsigned int globalID = r_x;
	//unsigned int pixelID = (int)globalID/(params.random_array_size*1.0f);
	//unsigned int rayID = globalID%params.random_array_size;
	unsigned int pixelID = globalID%(params.width*params.height);
	unsigned int rayID = (int) globalID/(params.width*params.height*1.0f);
	//unsigned int ray_index = r_x*paramsp->raysPerPixel
	//printf("source_pixels[1]: %d\n", source_pixels[1]);

	//printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
	*/
	/* 
	// Don't need sindices and svalues for the moment. can use shared memory for something else.

	if (threadIdx.x == 0 && threadIdx.y == 0){
	  random_number = random_array[rayID];
	 
	  threadCount = 0;
		for(int i = 0; i < blockDim.x*blockDim.y*num_passes; i++) {
		  sindices[i] = 0 ;
		  svalues[i] =  make_float3(0,0,0);
		}
	 
	  
	}
	*/
	__syncthreads();

	//atomicAdd(threadCount,1);
	//threadCount = threadCount+1;
	//printf("Thread Count: %d\n",threadCount);


	/*
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;  
	*/

	/*
	// Launch Photon if the corresponding pixel on the source image is white
	if(source_pixels[r_x]==0)
	{
	   //printf("k: %d, source_pixels[k]: %d\n",k,source_pixels[r_x]);
	   //paramsp->num_zero = paramsp->num_zero+1;
	   //printf("num_zero: %d\n",paramsp->num_zero);
	   return;
	}

	*/
	float3 pos, dir;
	//pos_c = params.pos_c;  // co-ordinates of the center of the dot card or BOS texture in physical space

	float3 accum = make_float3(0.0,0.0,0.0);
	float phase_shift = 0;
	  
	/*
	float pos_x = float(win_x%window_width)/float(window_width) - .5f;
	float cell_width = 1.0f/float(window_width);
	float cell_height = 1.0f/float(window_height);
	float pos_y = float((params.height-win_y)%window_height)/float(window_height) - .5f; // if reading from a background

	//float3 pos_i = pos_c + make_float3(1,0,0)*pos_x + make_float3(0,1,0)*pos_y;
	float3 pos_i = pos_c + make_float3(pos_x+cell_width,pos_y+cell_height,0);
	pos = pos_i;
	*/


	// Pinhole
	float c = params.c; // principal distance (orig 2.0)
	float3 pinhole_pos = params.camera_pos + params.camera_z*c;

	int3 particleIdx;
	particleIdx.x = threadIdx.x + blockIdx.x*blockDim.x ;
	particleIdx.y = threadIdx.y + blockIdx.y*blockDim.y ;
	particleIdx.z = blockIdx.z;

	int global_particleId = particleIdx.x + particleIdx.y*gridDim.x*blockDim.x + particleIdx.z*gridDim.y*blockDim.y*gridDim.x*blockDim.x; 


	// Generate random co-ordinates using curand
	unsigned int seed_offset = params.num_particles;
	float random_x = random_single(global_particleId) - 0.5f;
	float random_y = random_single(global_particleId + seed_offset) - 0.5f;
	float random_z = random_single(global_particleId + seed_offset*2)*0.2 - 0.1f;

	float disp_x = 0.001;
	float disp_y = 0.001;
	float disp_z = 0.001;

	/*
	// Generate Particle co-ordinates on a lattice
	float random_x = (particleIdx.x+0.5f)/(gridDim.x*blockDim.x) - 0.5f;
	float random_y = (particleIdx.y+0.5f)/(gridDim.y*blockDim.y) - 0.5f;
	float random_z = (particleIdx.z+0.5f)/(gridDim.z*blockDim.z) - 0.5f;
	*/
	pos = make_float3(random_x+disp_x,random_y+disp_y,random_z+disp_z); 

	// find vector that connects the source pixel to the pinhole
	dir = normalize(pinhole_pos-pos);
	if(dir.z < 0.0f)
		return;



	/*
	float random_x = params.pitch_random*random_number_1*cos(2*M_PI*random_number_2);
	float random_y = params.pitch_random*random_number_1*sin(2*M_PI*random_number_2);
	*/
	//float random_number_1 = 0.1; float random_number_2 = 0.2;
	//float3 random_pos = params.camera_x*random_x + params.camera_y*random_y + pinhole_pos; 
	//dir = normalize(random_pos-pos);

	if(!IntersectWithVolume(pos, dir, params.min_bound, params.max_bound)) 
		return;  

	float3 normal = {0.f,0.0f,-1.f};
		
  float old_index = 1.0;
	size_t DELTA = 1;
	int i = 0;
	int insideBox = 1;

	// Trace Ray through volume
	while(insideBox==1)
	{
			
    i = i+1;
    if(i>0)
		  pos = pos + dir*params.stepSize/old_index;
		float3 offset = pos-min_bound;
		float3 lookupfn = offset*lookup_scale; // normalized lookup

		float3 lookup = {static_cast<float>(lookupfn.x*params.data_width), static_cast<float>(lookupfn.y*params.data_height), static_cast<float>(lookupfn.z*params.data_depth)};
		// the curly brace on the previous line is for the lookup command, not the for loop

	  if(pos.x <= min_bound.x || pos.y <= min_bound.y || pos.z <= min_bound.z || pos.x >= max_bound.x || pos.y >= max_bound.y || pos.z >= max_bound.z )
    {
		   //printf("Ray has gone outside the volume. pos :%f, %f, %f\n", pos.x, pos.y, pos.z);
			break;
		}
		
	  if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA || lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA)
	      continue;
		
	  float4 val = tex3D(tex_data, lookup.x, lookup.y, lookup.z)*params.dataScalar;
		        
    val.w += 1.0; //TODO: should build this into main.cpp?

	  normal = make_float3(val.x,val.y,val.z);
	  float del_x = 1.0f/params.size_x;
	  normal = normal*1.0f/(2.0f*del_x); 
	  old_index = val.w;

	  //#if !LINE_OF_SIGHT
	  dir = dir + params.stepSize*normal;
	  phase_shift += val.w - 1.0;

	}

	// Find location of intersection of the ray with the pinhole plane
	pos = rayPlaneIntersection(pinhole_pos,params.camera_z,pos, dir);
	if(isnan(pos.x)+isnan(pos.y)+isnan(pos.z))
		return;
	
	// if the intersection point lies within the pitch of the pinhole, then allow the ray to pass through
	float pitch_pinhole = params.pitch;
	float3 dr = pos - pinhole_pos;
	float dx = length(proj3(dr,params.camera_x));
	float dy = length(proj3(dr,params.camera_y));

	if(length(dr)>pitch_pinhole)
 	  return;

	// Calculating the final position of the ray with respect to the camera plane 

	pos = rayPlaneIntersection(params.camera_pos,params.camera_z,pos,dir); 
	if(isnan(pos.x)+isnan(pos.y)+isnan(pos.z))
	  return;

	// if the intersection point lies within the camera sensor, then record the ray
	dr = pos - params.camera_pos;
	dx = abs(dot(dr,params.camera_x));
	dy = abs(dot(dr,params.camera_y));

	if(dx > params.camera_width/2.0 || dy > params.camera_width/2.0)
		return;

	float3 corner_offset = pos-params.camera_corner;
	float3 xoffset = proj3(corner_offset, params.camera_x);
	float3 yoffset = proj3(corner_offset, params.camera_y);

	// Convert location of the interseciton point to pixels
	// NOTE : Camera axes are such that the image is automatically inverted to appear as what you would see
	unsigned int w_x = length(xoffset)*params.width-1;
	unsigned int w_y = length(yoffset)*params.height-1;
	unsigned int win_index = w_y*params.width + w_x;

	// If the pixel location lies within the sensor of the camera, record the ray
	if (w_x <= params.width && w_y <= params.height)
	{
		// temporary fix for out_pixels
		//out_pixels[win_index] = out_pixels[win_index] + 255; //rgbToInt(255.0,0.0,0.0);

		out_pixels[win_index] = 65355;
		
		// Updating neighborhood elements to make the dot look bigger
		int left = w_y*params.width + w_x - 1;
		int right = w_y*params.width + w_x + 1;
		int top = (w_y+1)*params.width + w_x;
		int bottom = (w_y-1)*params.width + w_x;

		out_pixels[left] = 65355;
	 	out_pixels[right] = 65355;
	 	out_pixels[top] = 65355;
    out_pixels[bottom] = 65355;
	   
	 } 
	 

             
}

#endif // KERNEL_RENDER_H
