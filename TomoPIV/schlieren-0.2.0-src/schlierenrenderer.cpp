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
//#include "cutil.h"
//#include "cutil_math.h"
#include <cuda.h>
#include "cuda_gl_interop.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <fstream>
using namespace std;
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string>
#include <sstream>

extern "C" void Host_Render( RenderParameters* paramsp);
extern "C" void Host_Init(RenderParameters* paramsp);
extern "C" void Host_Clear(RenderParameters* paramsp);
extern "C" void Host_Kill();


float3 operator/(const float3 &a, const float &b) {

return make_float3(a.x/b, a.y/b, a.z/b);

}

float3 operator*(const float3 &a, const float &b) {

return make_float3(a.x*b, a.y*b, a.z*b);

}

float3 operator+(const float3 &a, const float3 &b) {

return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}

float3 operator-(const float3 &a, const float3 &b) {

return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

float3 operator-(const float3 &a) {

return make_float3(-a.x, -a.y, -a.z);

}

float3 normalize(const float3&a) {
    float m = sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
    return a/m;
}
float3 cross(const float3&a, const float3&b) {
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

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
deleteFramebuffer( GLuint* fbo)
{
  //    glDeleteFramebuffersEXT(1, fbo);
  //
  //    *fbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
createDepthBuffer(GLuint* depth, unsigned int size_x, unsigned int size_y)
{
  //    // create a renderbuffer
  //    glGenRenderbuffersEXT(1, depth);
  //    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth);
  //
  //    // allocate storage
  //    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, size_x, size_y);
  //
  //    // clean up
  //    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
  void
deleteDepthBuffer(GLuint* depth)
{
  //    glDeleteRenderbuffersEXT(1, depth);
  //
  //    *depth = 0;
}

  SchlierenRenderer::SchlierenRenderer()
: _initialized(false), _rot_x(0), _rot_y(0), _rot_z(0)
{
  _params.passes = 0;
  _params.inputFilename = "../inputParameters.txt";
  readParametersfromFile(_params.inputFilename);  

//  _params.width = 1024;  // default is 700 - use multiples of 16 to avoid noise at the edge
//  _params.height = 1024; // default is 700 - use mulitples of 16 to avoid noise at the edge
  int num_pixels = _params.width*_params.height;
  
  _params.inout_rgb = NULL;
  _params.out_rgb = NULL;

  _params.min_bound = make_float3(-.5,-.5,-.5);
  _params.max_bound = make_float3(.5,.5,.5);
   
  updateCamera();
  _params.raysPerPixel = 1;
  _params.random_array_size = 1; // number of light rays emitted for EACH light source on the BOS pattern    
  _params.num_zero = 0;
  _params.numRenderPasses = 1;

  _params.cutoff = CUTOFF_NONE;
  _params.cutoff_dirty = true;
  _params.data = NULL;
  _params.projectionDistance = 0.0;// default is 0.4f, turb channel : 0.0f
  _params.stepSize = 0.1;
  _params.threadSafe = false;
  _params.dataScalar = 1.0f; // default is 1.0f, turb channel : 1000.0f
  _params.cutoffScalar = 0.001f;  // default is 5000.0f, turb channel : 0.001f
  _params.useOctree = false;
  _params.useRefraction = false;
  _params.inout_rgb = new float4[_params.width*_params.height];
  _params.num_particles = 1e6;
  float3 camera_pos, camera_x, camera_y, camera_z, center;
//  float rot_x = 0, rot_y = 0, rot_z = 0;
  _center = make_float3(0,0,0);

  /*
  // Initialize the lightfield
  _params.lightfieldp = new LightField[num_pixels*_params.random_array_size];
  
  for (int i = 0; i<num_pixels*_params.random_array_size; i++){
    _params.lightfieldp[i].pos = new float3[_params.raysPerPixel];
    _params.lightfieldp[i].pos = make_float3(0,0,0);
    _params.lightfieldp[i].dir = new float3[_params.raysPerPixel];
    _params.lightfieldp[i].dir = make_float3(0,0,0);
  }
  */
 


}

void SchlierenRenderer::readParametersfromFile(string filename)
{
  //string filename = "inputParameters.txt";
  //string filename = "try1.txt";
  int a, b;
  FILE *inFile;
  inFile = fopen(filename.c_str(),"r");

  char line[1024];
  char str[10];
  int num; float f_num; float3 f3_num;
  int i = 1;

  printf("************************ Input Parameters ******************\n");
  // Read the first three lines which don't contain any parameters
  fgets(line,1024,inFile);
  fgets(line,1024,inFile);
  fgets(line,1024,inFile);

  // Read the image height
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %d",str,&num); 
  printf("%s %d\n", str,num);
  _params.height = num;

  // Read the image width
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %d",str, &num); 
  printf("%s %d\n", str,num);
  _params.width = num;
 
  // Read Distance of center of camera from the origin
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params.R = f_num;
 
  // Read Orientation of camera axes with respect to the world (Radians)
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params._rot_x = f_num;

  // Read Orientation of camera axes with respect to the world (Radians)
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params._rot_y = f_num;

  // Read Orientation of camera axes with respect to the world (Radians)
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params._rot_z = f_num;

  // Read Principal Distance - distance of pinhole from center of camera plane
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params.c = f_num;

  // Read Pitch or Aperture of the pinhole
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params.pitch = f_num;
  
  // Read width of the camera sensor
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params.camera_width = f_num;

  // Read height of the camera sensor
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f_num); 
  printf("%s %f\n", str,f_num);
  _params.camera_height = f_num;
  // Read X Location of center of BOS Target in world co-ord
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f3_num.x); 
  printf("%s %f \n", str,f3_num.x);
  _params.pos_c.x = f3_num.x;

  // Read Y Location of center of BOS Target in world co-ord
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f3_num.y); 
  printf("%s %f \n", str,f3_num.y);
  _params.pos_c.y = f3_num.y;

  // Read Z Location of center of BOS Target in world co-ord
  fgets(line,1024,inFile);
  sscanf(line, "%s %*s %f",str, &f3_num.z); 
  printf("%s %f \n", str,f3_num.z);
  _params.pos_c.z = f3_num.z;

  printf("************************************************************\n");
  fclose(inFile);
  
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
 printf("Hello");
  printf("cutoff: %d\n",_params.cutoff);
  Host_Render(&_params);
  printf("passes: %d\n",_params.passes);

}

void SchlierenRenderer::rotate(float x, float y)
{
  _rot_x += x;
  _rot_y += y;
  updateCamera();
}

void SchlierenRenderer::updateCamera()
{
  // This function rotates the camera by specified angles and calculates the orientation of the new camera axes in world co-ordinate system

  float3 _center = make_float3(0,0,0);  // origin in world co-ordinates
  float R = _params.R;

  _rot_x = _params._rot_x; // rotation in radians about global X
  _rot_y = _params._rot_y; // rotation in radians about global Y
  _rot_z = _params._rot_z; // rotation in radians about global Z

  // Evaluate the elements in the Eulerian rotation matrix
  float A11, A12, A13, A21, A22, A23, A31, A32, A33;
  float phi, theta, psi;  // phi - rotation about global X, theta - rotation about global Y, psi - rotation about Z

  phi = _rot_x; theta = _rot_y; psi = _rot_z;

  // Calculate components of EXTRINSIC rotation matrix (From Wikipedia page on Rotational Formalisms in three dimensions)
  A11 = cos(theta)*cos(psi); A12 = cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi); A13 = sin(phi)*sin(psi)-cos(phi)*sin(theta)*cos(psi);
  A21 = -cos(theta)*sin(psi); A22 = cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi); A23 = sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi);
  A31 = sin(theta) ; A32 = -sin(phi)*cos(theta); A33 = cos(phi)*cos(theta);
  
  // Transform the camera axis using an Eulerian rotation  
  _params.camera_x = make_float3(A11, A12, A13);
  _params.camera_y = make_float3(A21, A22, A23);
  _params.camera_z = make_float3(A31, A32, A33);

  // Set Position of Camera center and camera corners in world coordinate system
  // set the camera position such that the camera z connects the camera center to the origin
  //_params.camera_pos = _center + make_float3(R*sin(_rot_y),0,R*cos(_rot_y));
  _params.camera_pos = _center + _params.camera_z*-1.0*R;
  _params.camera_corner = _params.camera_pos - (_params.camera_x*.5+_params.camera_y*.5);
  _params.camera_pinhole_pos = _params.camera_pos + _params.camera_z*_params.c;

  // Display camera parameters to the user
  printf("************************************************************\n");
  printf("Camera Parameters: \n");
  printf("min_bound: %f, %f, %f\n", _params.min_bound.x, _params.min_bound.y, _params.min_bound.z);
  printf("max_bound: %f, %f, %f\n", _params.max_bound.x, _params.max_bound.y, _params.max_bound.z);
  printf("center: %f, %f, %f\n", _center.x, _center.y, _center.z);
  printf("camera_x: %f, %f, %f\n", _params.camera_x.x, _params.camera_x.y, _params.camera_x.z);
  printf("camera_y: %f, %f, %f\n", _params.camera_y.x, _params.camera_y.y, _params.camera_y.z);
  printf("camera_z: %f, %f, %f\n", _params.camera_z.x, _params.camera_z.y, _params.camera_z.z);
  printf("camera_pos: %f, %f, %f\n",_params.camera_pos.x, _params.camera_pos.y, _params.camera_pos.z);
  printf("camera_corner: %f, %f, %f\n", _params.camera_corner.x, _params.camera_corner.y, _params.camera_corner.z);
  printf("pinhole_pos: %f, %f, %f\n", _params.camera_pinhole_pos.x, _params.camera_pinhole_pos.y, _params.camera_pinhole_pos.z); 
  printf("************************************************************\n");
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
  float m = fmax(float(data_width), float(data_height));
  m = fmax(m, float(data_depth));
  float3 dataScale = make_float3(float(data_width)/m, float(data_height)/m, float(data_depth)/m);
  _params.min_bound = -dataScale/2.0f;
  _params.max_bound = dataScale/2.0f;

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
  Host_Kill();
}
            
                                                                                                                                                                                                                                                                            

