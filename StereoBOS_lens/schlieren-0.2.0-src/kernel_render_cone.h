#ifndef KERNEL_RENDER_H
#define KERNEL_RENDER_H

texture<float4, 3> tex_data;
texture<float, 3> tex_data2;
texture<float,1> tex_rand_phi;
texture<float,1> tex_rand_theta;
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


__global__ void kernel_render(float3* ray_pos, float3* ray_dir, RenderParameters* paramsp, float4* inout_pixels, unsigned int* out_pixels, float* random_array_theta,float*  random_array_phi, unsigned int* source_pixels,int *launchCount,int* receiveCount, int* threshCount)
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
   
   pos_c = make_float3(0,0,-0.5);   // co-ordinates of the center of the dot card or BOS texture in physical space
  
  
        atomicAdd(launchCount, 1);
        float3 accum = make_float3(0.0,0.0,0.0);
        float phase_shift = 0;
      

        float pos_x = float(win_x%window_width)/float(window_width) - .5f;
        float cell_width = 1.0f/float(window_width);
        float cell_height = 1.0f/float(window_height);
        float pos_y = float(win_y%window_height)/float(window_height) - .5f;
        
        //float3 pos_i = pos_c + make_float3(1,0,0)*pos_x + make_float3(0,1,0)*pos_y;
        float3 pos_i = pos_c + make_float3(pos_x+cell_width,pos_y+cell_height,0);
	pos = pos_i;
        //if(r_x == 62328)
	//  printf("r_x: %d, pos : %f, %f, %f\n", r_x, pos.x, pos.y, pos.z);

	// Pinhole
	float c = 2.0; // principal distance (orig 2.0)
	float3 pinhole_pos = params.camera_pos + params.camera_z*c;
	
	// find vector that connects the source pixel to the pinhole
        float3 dir_i = normalize(pinhole_pos-pos);
	dir = dir_i;
        //float err = 100;
 
        //*************************************** Generate a cone of light rays ********************************************
        
        // Define Cone Parameters  (MUST BE IN DECIMALS)
        float R = 1.0, phi_cone = 5.0/180.0*M_PI; // phi_cone is cone half angle in radians
        float del_phi = 0.5/180.0*M_PI, del_theta = 5.0/180.0*M_PI;	// angular interval (in radians) between two rays

	//int N_theta = 101, N_phi = 101; // Number of intervals for theta and phi
        
        // Define new co-ordinate system where the ray is one of the axes
        float3 dir_1 = dir;
        float3 dir_2 = normalize(cross(dir_1, make_float3(0,1,0)));
        float3 dir_3 = normalize(cross(dir_2, dir_1));
        
        int j,k; 
        float3 pt_global = make_float3(0,0,0);
	int N_phi = int(1+phi_cone/del_phi);
	int N_theta = int(1+2*M_PI/del_theta);
        
        //printf("N_phi: %d, N_theta: %d\n",N_phi, N_theta); 
	//float del_phi = phi_cone/(N_phi-1.0);
        //float del_theta = 2*M_PI/(N_theta-1.0);
	
	float thresh = 0.1*M_PI/180.0; // threshold for angle in radians
        float theta, phi;
        float x, y, z;
	float angle;
	//int flag = 0;
	//N_phi = 1; del_phi = 0;
	//N_theta = 1; del_theta = 0; 
     
     for(j = 0; j<N_phi; j++) 
     {
       phi = j*del_phi;
     
       for(k = 0; k<N_theta; k++)
       {
          // Generate direction vector for a line along the cone
          theta = -M_PI + k*del_theta;          
          x = R*__sinf(phi)*__cosf(theta);
          y = R*__sinf(phi)*__sinf(theta);
          z = R*__cosf(phi);
            
          pt_global = dir_1*z + dir_2*x + dir_3*y;
          dir = normalize(pt_global);
          pos = pos_i; 
	  //dir = dir_i; flag = 1;// remove thresholding when you have this line uncommented
          //dir = dir_1;	  
	  //printf("R,theta,phi, M_PI: %f, %f, %f, %f\n",R,theta,phi, M_PI);
          // Begin Ray Tracing
   	  //IntersectWithVolume(pos, dir, params.min_bound, params.max_bound);

			
          //pos = pos_i+dir*params.stepSize;
	  //pos = pos_i; dir = dir_i;
          if(!IntersectWithVolume(pos, dir, params.min_bound, params.max_bound)) 
		continue;  
	//pos = pos_i+dir*params.stepSize;
 
	  float3 normal = {0.f,0.0f,-1.f};
        
	  //int steps = 1.4f/(params.stepSize);
          float old_index = 1.0;
          size_t DELTA = 1;
	  int i = 0;
          int insideBox = 1;
	  //for(int i = 0; i < steps; ++i) 
          // Trace Ray through volume
	  while(insideBox==1)
	  {
             i = i+1;
	     if(i>0)
	     	pos = pos + dir*params.stepSize/old_index;
             float3 offset = pos-min_bound;
             float3 lookupfn = offset*lookup_scale; // normalized lookup

         float3 lookup = {static_cast<float>(lookupfn.x*params.data_width), static_cast<float>(lookupfn.y*params.data_height), static_cast<float>(lookupfn.z*params.data_depth)         };
        // the curly brace on the previous line is for the lookup command, not the for loop

           if(pos.x <= min_bound.x || pos.y <= min_bound.y || pos.z <= min_bound.z ||
               pos.x >= max_bound.x || pos.y >= max_bound.y || pos.z >= max_bound.z )
            {
	       //printf("Ray has gone outside the volume. pos :%f, %f, %f\n", pos.x, pos.y, pos.z);
		break;
	    }
        
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
	
	float3 l_pinhole = normalize(pinhole_pos-pos); // direction vector connecting the current position of the ray to the pinhole
	angle = acosf(dot(l_pinhole,dir)); // angle between two vectors in radians
        
        // if angle is sufficiently small, then stop generating any more rays for this pixel location
	if(angle>=-thresh && angle<=thresh) 
        {
	 atomicAdd(threshCount,1);
	//***************************** Calclating the final position of the ray with respect to the camera plane ********************

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
        //printf("pos: %f, %f, %f\n",pos);   
        // find the location of the intersection point on the camera plane
        float3 corner_offset = pos-params.camera_corner;
        float3 xoffset = proj3(corner_offset, params.camera_x);
        float3 yoffset = proj3(corner_offset, params.camera_y);

        // Convert location of the interseciton point to pixels
        unsigned int w_x = length(xoffset)*params.width-1;
        unsigned int w_y = length(yoffset)*params.height-1;
        unsigned int win_index = w_y*params.width + w_x;
	//printf("w_x: %d, w_y: %d\n", w_x, w_y);

        //win_index = w_x*params.height + w_y;
        
        // If the pixel location lies within the sensor of the camera, record the ray
	if (w_x <= params.width && w_y <= params.height)
	{
	  atomicAdd(receiveCount, 1);
	  //atomicAdd(offsetCount,length(offset));
          out_pixels[win_index] = 255; //rgbToInt(255.0,0.0,0.0);
	}
	 //printf("Angle: %f, pos: %f,%f,%f\n",angle, pos); 
	 break;
	}
       }
     }        
        //flag = 1;
 //     if(flag==0)
   //    return;


}


#endif // KERNEL_RENDER_H
