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





#define BLUR_DATA 0
#define GAUSSIAN 1  //guassian or mean


#include "opengl_include.h"
#include "cutil.h"
#include "cutil_math.h"
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "main.h"
#include "schlierenrenderer.h"
#include <fstream>
#include <float.h>
#include <assert.h>
//#include <mpi.h>
using namespace std;


float data_fudge = 1.0f;

SchlierenRenderer renderer;

void display()
{
  cout << "glut display\n";
  static int count = 0;
  
  std::cout << "On loop number " << count << "\n";

  if (count++ >= 1)
  {
    cout << "finished rendering\n";
    return;

  }
  printf("Hello");
  renderer.render();
 printf("Hello");
  glutPostRedisplay();
  glutSwapBuffers();
}

void keyboard( unsigned char key, int x, int y) {}
double last_x = 0.0;
double last_y = 0.0;
bool dirty_gradient = true;
bool alteringTF = false;
bool rotating = false;
void mouse(int button, int state, int x, int y) {
  if (button == GLUT_RIGHT_BUTTON) {
#if USE_TF
    if (state == GLUT_DOWN)
    {
      alteringTF = true;
      dirty_gradient = true;
      last_x = x;
      last_y = y;
    }
    else {
      //cudaMemcpy(d_TFBuckets, transferFunction->buckets, sizeof(float4)*transferFunction->numBuckets, cudaMemcpyHostToDevice);
      alteringTF = false;
    }
#endif
  }
  else if (button == GLUT_LEFT_BUTTON) {
    last_x = x;
    last_y = y;
    if (state == GLUT_DOWN) {
      rotating = true;
    }
    else {
      rotating = false;
      renderer.clear();
    }
  }
}
double rot_x = 0;
double rot_y = 0;
void motion(int x, int y)
{
#if USE_TF
  if (alteringTF) {
    for(int i =0 ; i < 100; i++) {
      int tx = last_x + (x - last_x)*(i/100.0f);
      int ty = last_y + (y - last_y)*(i/100.0f);
      if (tfRenderable->Contains(tx,window_height - ty))
        tfRenderable->ProcessMouse(tx,window_height -ty);
    }
    clear();
  }
#endif
  float scale = 0.00001;
  if (rotating) {
    rot_x += (x-last_x)*scale;
    rot_y += (y-last_y)*scale;

    //setCamera();
    renderer.rotate(rot_x, rot_y);
    renderer.updateCamera();

    //    printf("setting camera: %f %f %f\n", camera_pos.x, camera_pos.y, camera_pos.z);
    //    printf("magnitude of camera from center: %f\n", float(magnitude(minus(camera_pos, center))));
    //    printf("camera rotx roty: %f %f\n", rot_x, rot_y);
    renderer.clear();
  }
  last_x = x;
  last_y = y;
}
bool initGL()
{
  return true;
}

void reshape(int w, int h)
{
  renderer.setRenderSize(w,h);
}


/*template<class T, int N> convolve(T* data_in, T* data_out, float* kernel, float* kernel_width, size_t stride, size_t steps)
  {
  T* din = data_in;
  T* dout = data_out;
  for(size_t i = 0; i < steps; i++) {

  }
  }*/

#define UNIVERSALGASCONSTANT    8.314472                // J/(mol.K) = BOLTZMANNSCONSTANT * AVOGADROSNUMBER

// Ciddor's formulation
extern "C" double   AirRefractiveIndex(double temperature = 293.15, double pressure = 101325.0, double wavelength = 580.0, double relativeHumidity = 0.0, double co2ppm = 450.0);

// Simpler formulation
extern "C" double   AirRefractionIndex(double temperature = 293.15, double pressure = 101325.0, double wavelength = 580.0);
extern "C" void loadNRRD(DataFile* datafile, int data_min, int data_max);

int main(int argc, char** argv)
{
  bool record = false;
  float step_size = 0.1;
  int data_min = 0;
  int data_max = 1024;
  bool temp = false;
  bool press = false;
  bool convert = false;
  string filename = "/home/sci/brownlee/data/ExxonMobil/faults.nrrd";
  vector<string> files, tempFiles, pressFiles;
  DataFile* dataFiles[200];
 
  files.push_back(string(argv[1]));
  
  for(int file = 0; file < files.size(); file++) {
    dataFiles[file] = new DataFile();
    dataFiles[file]->filename = new char[256];
  }
  
  char** input_files;
  input_files = new char*[files.size()];
  for( int i = 0; i < files.size(); i++) {
    cout << "file: " << files[i] << endl;
    input_files[i] = new char[files[i].length()];
    strcpy(input_files[i], files[i].c_str());
    strcpy(dataFiles[i]->filename, input_files[i]);
    loadNRRD(dataFiles[i],data_min, data_max);
  }

  float* data = dataFiles[0]->data;
  int zrange = dataFiles[0]->sizez;

  // Create BOS Texture
  
  string filename_BOS = "../BOS_texture.bin";
   int k;
   int N = renderer._params.width*renderer._params.height;
    std::ifstream myFile1 (filename_BOS.c_str(), std::ios::in | std::ios::binary);
    renderer._params.source_rgb = (unsigned int*) malloc (N*sizeof(unsigned int));
    renderer._params.out_rgb = (unsigned int*) malloc (N*sizeof(unsigned int));
    printf("Reading from file\n");
    for(k = 0; k<N; k++)
   {
     myFile1.read ((char*)&renderer._params.source_rgb[k], sizeof(unsigned int));
     renderer._params.out_rgb[k]=0;
    //printf("source_rgb[%d]: %d\n",k,renderer._params.source_rgb[k]);
   }
   //myFile1.read((char*)&renderer._params.source_rgb, N*sizeof(unsigned int));
   myFile1.close();

  printf("Finished Reading from file\n");

  
  cout << "setting up renderer\n";
      
  renderer.setData(data, dataFiles[0]->sizex,dataFiles[0]->sizey,zrange);
  renderer.setStepSize(step_size);

  //renderer.setFilter(new SchlierenImageCutoff(cutoffData));
  //renderer.setFilter(new SchlierenInterforemetryCutoff());
  cout << "setting filter\n";
  //renderer.setFilter(new SchlierenPositiveHorizontalKnifeEdgeCutoff());
   renderer.setFilter(new SchlierenShadowgraphCutoff());
  cout << "setting image filter\n";
  renderer.setImageFilter(new ImageFilter());
  cout << "setting up glut\n";

  glutInit( &argc, argv);
  glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize( renderer.getWidth(), renderer.getHeight());
  glutCreateWindow( "CD CUDA Schlieren");

  // initialize GL
  if( false == initGL()) {
    exit(1);
  }

  // register callbacks
  glutDisplayFunc( display);
  glutKeyboardFunc( keyboard);
  glutMouseFunc( mouse);
  glutMotionFunc( motion);
  glutReshapeFunc(reshape);

  
  string arg = "-file";
  cout<<"You have chosen to write data to "<<arg<<"\n";
  
  if(arg=="-screen")
  {
    printf("Entering Main Loop\n");
    cout << "mainloop\n";

    // start rendering mainloop
     glutMainLoop();
  
    cout << "mainloop done\n";
  }

  
  int i;
  for(i = 1; i<=1; i++)
  {
    renderer.render();
  }

// Access the RGB values
 unsigned int* out_rgb = renderer._params.out_rgb;

  int width = renderer._params.width;
  int height = renderer._params.height;
  int num_pixels = width * height;
  printf("num_pixels: %d\n",num_pixels);
  printf("size of unsigned int: %d\n",sizeof(unsigned int));

   string filename2(argv[1]);
   cout<<filename2<<"\n";
   int len = filename2.length()-5;
   cout<<"len: "<<len<<"\n";
   string arg2 = filename2.substr(0,len);
   arg2.append(".bin");
   cout<<"arg2: "<<arg2<<"\n";
   arg2 = "../BOS_NS.bin";
   //int k;
   int sum_out_rgb = 0;
   for(k = 0;k<num_pixels;k++)
     sum_out_rgb = sum_out_rgb+renderer._params.out_rgb[k];
   
   printf("sum_out_rgb before writing to file: %d\n",sum_out_rgb);
   
   std::ofstream myFile (arg2.c_str(), std::ios::out | std::ios::binary);
    std::ofstream sourceFile("../source_pixels.bin", std::ios::out | std::ios::binary);
   for(k = 0; k<num_pixels; k++)
   {
    if(renderer._params.out_rgb[k]>255)
      renderer._params.out_rgb[k]=0;
     myFile.write ((char*)&renderer._params.out_rgb[k], sizeof(unsigned int));
    sourceFile.write((char*)&renderer._params.source_rgb[k], sizeof(unsigned int));
   }
    myFile.close();
    sourceFile.close();

  printf("Finished Writing to file\n");

  renderer.clear();
  delete out_rgb;
  //delete renderer;
  // MPI_Init(&argc, &argv);
  int rank, size;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //   MPI_Comm_size(MPI_COMM_WORLD, &size);
  rank = 0;
  size = 1;
  // run(argc, argv, rank, size, 0, 0, 0, input_files, files.size(), data_min, data_max, dataFiles, step_size, record);
  //    MPI_Finalize();
  return 0;
}





extern "C"
{
  void loadNRRD(DataFile* datafile, int data_min, int data_max)
  {

    printf("loading file %s : ", datafile->filename);
    Nrrd* nrrd = nrrdNew();
    if(nrrdLoad(nrrd, datafile->filename, 0)) {
      char* err=biffGetDone(NRRD);
      cerr << "Failed to open \"" + string(datafile->filename) + "\":  " + string(err) << endl;
      exit(__LINE__);
    }
    int sizex, sizey, sizez;
    sizex = nrrd->axis[0].size;
    sizey = nrrd->axis[1].size;
    sizez = nrrd->axis[2].size;
    if (data_max > sizez)
      data_max = sizez;
    if (sizez > (data_max-data_min))
      sizez = data_max-data_min;
    printf(" size: %f %f %f ", float(sizex), float(sizey), float(sizez));
    float* data = new float[sizex*sizey*sizez];
    float min = FLT_MAX;
    float max = -FLT_MAX;
    float* dataNrrd = (float*)nrrd->data;
    float* datai = data;
    for(int i = 0; i < sizex; i++) {
      for(int j = 0; j < sizey; j++) {
        for( int k = 0; k < sizez; k++) {
          *datai = (*dataNrrd)*data_fudge;

          if (*datai > max)
            max = *datai;
          if (*datai < min)
            min = *datai;
          datai++;
          dataNrrd++;

        }
      }
    }


    datafile->data = data;
    datafile->sizex = sizex;
    datafile->sizey = sizey;
    datafile->sizez = sizez;
    nrrdNuke(nrrd);
    printf("  ...done\n");
  }


  // --------------------------------------------------------------------------------------
  // -- Return the compressibility
  // -- temperature                               : Kelvin
  // -- pressure                                  : Pascal
  // -- waterVaporMolarFraction   : [0-1]
  // --------------------------------------------------------------------------------------
  double Compressibility(double temperature, double pressure, double waterVaporMolarFraction)
  {
    double a0, a1, a2, b0, b1, c0, c1, d, e, z, pt, tC;

    a0 = 1.58123e-6;
    a1 = -2.9331e-8;
    a2 = 1.1043e-10;
    b0 = 5.707e-6;
    b1 = -2.051e-8;
    c0 = 1.9898e-4;
    c1 = -2.376e-6;
    d  = 1.83e-11;
    e  = -0.765e-8;

    pt = pressure / temperature;
    tC = temperature - 273.15;      // Temperature in Celcius

    z = 1.0 + pt * (pt * (d + e*waterVaporMolarFraction*waterVaporMolarFraction)
        - (a0 + (a1 + a2*tC)*tC + ((b0 + b1*tC) + (c0 + c1*tC)*waterVaporMolarFraction) * waterVaporMolarFraction));

    return z;
  }


  // --------------------------------------------------------------------------------------
  // -- Compute the dryAirComponent and waterVaporComponent of the density
  // -- temperature                               : Kelvin
  // -- pressure                                  : Pascal
  // -- waterVaporMolarFraction   : [0-1]
  // -- co2ppm                                    : parts per million
  // --------------------------------------------------------------------------------------
  void Density(double temperature, double pressure, double waterVaporMolarFraction, double co2ppm, double * dryAirComponent, double * waterVaporComponent)
  {
    double pzrt, Ma, Mw, z;

    Mw = 0.018015;                                                                  // Molar mass of water vapor
    Ma = 0.0289635 + 12.011e-9 * (co2ppm - 400.0);  // Molar mass of dry air containing co2 ppm

    z = Compressibility(temperature, pressure, waterVaporMolarFraction);

    pzrt = pressure / (z * UNIVERSALGASCONSTANT * temperature);

    if (dryAirComponent)
      *dryAirComponent                = pzrt * Ma * (1.0 - waterVaporMolarFraction);

    if (waterVaporComponent)
      *waterVaporComponent    = pzrt * Mw * (      waterVaporMolarFraction);
  }


  // --------------------------------------------------------------------------------------
  // -- Return the (refractive index of air - 1) for the given parameters
  // -- temperature               : Kelvin
  // -- pressure                  : Pascal
  // -- wavelength                : nanometer
  // -- relativeHumidity  : [0-1]
  // -- co2ppm                    : parts per million
  // --------------------------------------------------------------------------------------
  double AirRefractiveIndex(double temperature, double pressure, double wavelength, double relativeHumidity, double co2ppm)
  {
    // Saturation vapor pressure of water vapor in air
    double svp = exp((1.2378847e-5*temperature - 1.9121316e-2)*temperature + 33.93711047 - 6.3431645e3/temperature);

    // Enhancement factor of water vapor in air
    double f, tC = temperature - 273.15;
    f = 1.00062 + 3.14e-8*pressure + 5.6e-7*tC*tC;

    // Molar fraction of water vapor
    double xw = relativeHumidity * f * svp / pressure;

    double paxs, pws, pa, pw;
    Density(     288.15, 101325.0, 0.0, co2ppm, &paxs, NULL);       // Density of standard dry air
    Density(     293.15,   1333.0, 1.0, co2ppm,  NULL, &pws);       // Density of standard water vapor
    Density(temperature, pressure,  xw, co2ppm,   &pa,  &pw);       // Density of moist air

    double waveNumber, waveNumber2;
    waveNumber = 1000.0 / wavelength;       // nanometer to micrometer
    waveNumber2 = waveNumber * waveNumber;

    // Refractivity of standard air (15 C, 101325 Pascal, 0% humidity, 450 ppm of CO2)
    double nas1 = (5792105.0 / (238.0185  - waveNumber2) + 167917.0 / (57.362  - waveNumber2)) * 1.0e-8;

    // Refractivity of standard air with co2 ppm
    double naxs1 = nas1 * (1.0 + 0.534e-6 * (co2ppm - 450.0));

    // Refractivity of standard water vapor (20 C, 1333 Pascal, 100% humidity)
    double nws1 = 1.022e-8 * (295.235 + (2.6422 - (0.03238 + 0.004028*waveNumber2) * waveNumber2) * waveNumber2);

    return naxs1 * pa / paxs + nws1 * pw / pws;
  }


  // --------------------------------------------------------------------------------------
  // -- Return (refractiveIndex - 1)
  // -- temperature               : Kelvin
  // -- pressure                  : Pascal
  // -- wavelength                : meter
  // --------------------------------------------------------------------------------------
  double AirRefractionIndex(double temperature, double pressure, double wavelength)
  {
    double tempC, sigma, index;

    tempC = temperature - 273.15;
    sigma = 1.0e-6 / wavelength;
    index = 0.0472326 / (173.3 - sigma * sigma);
    index = index * pressure * (1.0 + pressure * (60.1 - 0.972 * tempC) * 1e-10) / (96095.43 * (1.0 + 0.003661 * tempC));

    return index;
  }

}