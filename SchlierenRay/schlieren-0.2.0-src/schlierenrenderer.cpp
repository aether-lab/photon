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



#include "schlierenrenderer.h"

#include <stdio.h>

#include "opengl_include.h"
#include <cuda.h>
#include "cuda_gl_interop.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <fstream>
#include <math.h>
#include <float.h>

using namespace std;

extern "C" void Host_Render( RenderParameters* paramsp);
extern "C" void Host_Init(RenderParameters* paramsp);
extern "C" void Host_Clear(RenderParameters* paramsp);
extern "C" void Host_Kill();

// following lines perform operator overloading
float3 operator/(const float3 &a, const float &b)
{
    return make_float3(a.x/b, a.y/b, a.z/b);
}

float3 operator*(const float3 &a, const float &b) 
{
    return make_float3(a.x*b, a.y*b, a.z*b);
}

float3 operator+(const float3 &a, const float3 &b) 
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

float3 operator-(const float3 &a, const float3 &b) 
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

float3 operator-(const float3 &a) 
{
    return make_float3(-a.x, -a.y, -a.z);
}

float3 normalize(const float3&a) 
{
    float m = sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
    return a/m;
}

float3 cross(const float3&a, const float3&b) 
{
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

// pbo and fbo variables
GLuint pbo_out, pbo_dest;
GLuint fbo_source;
GLuint tex_source;

// (offscreen) render target
// fbo variables
GLuint framebuffer;
GLuint tex_screen;
GLuint depth_buffer;


// display image to the screen as textured quad
void SchlierenRenderer::displayImage()
{
  // render a screen sized quad
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode( GL_MODELVIEW);
  glLoadIdentity();

  glViewport(0, 0, _params.width, _params.height);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
  glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
  glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
  glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);

}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y)
{
  // create a texture
  glGenTextures(1, tex_name);
  glBindTexture(GL_TEXTURE_2D, *tex_name);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // buffer data
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
deleteTexture(GLuint* tex)
{
  glDeleteTextures(1, tex);
  *tex = 0;
}

SchlierenRenderer::SchlierenRenderer()
: _initialized(false), _rot_x(0), _rot_y(0)
{
  _params.passes = 0;
  //_params.tex_data = NULL;
  _params.width = 2048;  // default is 700
  _params.height = 512; // default is 700
  _params.inout_rgb = NULL;
  _params.out_rgb = NULL;
  updateCamera();
  _params.raysPerPixel = 1;
  _params.numRenderPasses = 1;
  _params.cutoff = CUTOFF_NONE;
  _params.cutoff_dirty = true;
  _params.data = NULL;
  _params.projectionDistance = 0.0;// default is 0.4f, turb channel : 0.0f
  _params.stepSize = 0.1;
  _params.threadSafe = false;
  _params.useOctree = false;
  _params.useRefraction = false;
  _params.inout_rgb = new float4[_params.width*_params.height];

  float3 camera_pos, camera_x, camera_y, camera_z, center = _params.max_bound/2.0;
  float rot_x = 0, rot_y = 0;

  _center = make_float3(0,0,0);
}

SchlierenRenderer::~SchlierenRenderer()
{
  Host_Kill();
}

void SchlierenRenderer::init()
{
  _initialized=true;
  Host_Init(&_params);

}

void SchlierenRenderer::render()
{
  if (!_initialized)
    init();

  // scale physical volume to fit inside a 1x1x1 cube
  scaleVolume();
  // shift volume to ensure that center is at 0,0,0
  shiftVolume();

  Host_Render(&_params);
  printf("passes: %d\n",_params.passes);

}

void SchlierenRenderer::rotate(float x, float y)
{
  _rot_x += x;
  _rot_y += y;
  updateCamera();
}

void SchlierenRenderer::scaleVolume()
{
// this function scales the physical volume to fit inside a 1x1x1 cube AND maintaining the aspect ratio

	// find the extent of the volume
	float3 xyz_extent = _params.data_max_bound - _params.data_min_bound;
	
	// find extent along each co-ordinate axis
	float x_extent = xyz_extent.x;
	float y_extent = xyz_extent.y;
	float z_extent = xyz_extent.z;

	// find the max extent which needs to be scaled down to 1
	float scale_factor = max(x_extent, y_extent);
	scale_factor = max(scale_factor, z_extent);
	
	scale_factor = 0.3*scale_factor; // temporary
	// scale down the volume
	_params.min_bound = _params.data_min_bound/scale_factor;
	_params.max_bound = _params.data_max_bound/scale_factor;

	// display the results
	printf("Scaling the Physical Volume\n");
	printf("scale factor: %f\n", scale_factor);
	printf("min_bound: %f, %f, %f\n", _params.min_bound.x, _params.min_bound.y, _params.min_bound.z);
	printf("max_bound: %f, %f, %f\n", _params.max_bound.x, _params.max_bound.y, _params.max_bound.z);

}


void SchlierenRenderer::setData(float* data, int data_width, int data_height, int data_depth)
{

    printf("SchlierenRenderer::setData(%d, %d, %d, %d)\n", (u_int64_t)data, data_width, data_height, data_depth);
  _params.data_min = FLT_MAX;
  _params.data2 = data;
  //compute gradient
  _params.data = new float4[data_width*data_height*data_depth];
  for(size_t z = 0; z < data_depth; z++) {
    for(size_t y = 0; y < data_height; y++) {
      for(size_t x = 0; x < data_width; x++) {
        size_t DELTA = 1;
        float3 lookup = {x,y,z};
        if (lookup.x < DELTA || lookup.y < DELTA || lookup.z < DELTA ||
            lookup.x >= data_width-DELTA || lookup.y >= data_height -DELTA || lookup.z >=data_depth-DELTA)
          continue;
        float3 sample1, sample2;
        lookup = make_float3(x-1,y,z);
        sample1.x = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        lookup = make_float3(x+1,y,z);
        sample2.x = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];

        lookup = make_float3(x,y-1,z);
        sample1.y = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        lookup = make_float3(x,y+1,z);
        sample2.y = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];

        lookup = make_float3(x,y,z-1);
        sample1.z = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        lookup = make_float3(x,y,z+1);
        sample2.z = data[size_t(lookup.z*data_width*data_height + lookup.y*data_width + lookup.x)];
        float3 normal;
        normal = sample1 - sample2;
        float4& datap = _params.data[size_t(z*data_width*data_height + y*data_width + x)];
        datap.x = normal.x;
        datap.y = normal.y;
        datap.z = normal.z;
        datap.w = data[size_t(z*data_width*data_height + y*data_width + x)];
        if (datap.w < _params.data_min)
          _params.data_min = datap.w;
      }
    }
  }

  _params.data_width = data_width;
  _params.data_height = data_height;
  _params.data_depth = data_depth;
}

void SchlierenRenderer::setFilter(SchlierenCutoff* filter)
{
  filter->Setup(_params);
}

void SchlierenRenderer::clear()
{
   Host_Clear(&_params);
  _params.passes = 0;
}

void SchlierenRenderer::shiftVolume()
{
// this function shifts the scaled volume to ensure that the center of the volume is 
// at 0,0,0

	// find the current center of the volume
	float3 center_volume = (_params.min_bound + _params.max_bound)*0.5;

	// find the amount by which it needs to be shifted to have the center at 0,0,0
	float3 shift_required = center_volume*-1.0;

	// perform the shifting
	_params.min_bound = _params.min_bound + shift_required;
	_params.max_bound = _params.max_bound + shift_required;

	// display the results
	printf("Shifting the scaled Physical Volume\n");
	printf("shift required: %f, %f, %f\n", shift_required.x, shift_required.y, shift_required.z);
	printf("min_bound: %f, %f, %f\n", _params.min_bound.x, _params.min_bound.y, _params.min_bound.z);
	printf("max_bound: %f, %f, %f\n", _params.max_bound.x, _params.max_bound.y, _params.max_bound.z);

}


void SchlierenRenderer::updateCamera()
{

  float3 center = _params.max_bound/2.0f;

  _params.camera_pos.z = center.x + cos(_rot_y)*cos(_rot_x)*5.0;
  _params.camera_pos.y = center.y + sin(_rot_y)*5.0;

  _params.camera_pos.x = center.z + sin(_rot_x)*5.0;

  _params.camera_z = normalize(center-_params.camera_pos);

  _params.camera_y = make_float3(0,1,0);
  _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
  _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));


  _params.camera_pos.z = center.z + cos(_rot_x)*5.0;
  _params.camera_pos.x = center.x + sin(_rot_x)*5.0;
  _params.camera_z = normalize(center-_params.camera_pos);

  _params.camera_y = make_float3(0,1,0);
  _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
  _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));



  _params.camera_pos.z = _center.z + cos(_rot_y)*5.0;
  _params.camera_pos.y = _center.y + sin(_rot_y)*5.0;
  _params.camera_z = normalize(_center-_params.camera_pos);

  _params.camera_y = make_float3(0,1,0);
  _params.camera_x = normalize(cross(_params.camera_y, _params.camera_z*-1.0f));
  //    camera_x = set_float3(1,0,0);
  _params.camera_y = normalize(cross(_params.camera_x, _params.camera_z));



  _params.camera_corner = _params.camera_pos-(_params.camera_x*.5+_params.camera_y*.5);
}

