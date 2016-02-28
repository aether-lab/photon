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
#include <tiffio.h>
#include <cstring>
#include <stdint.h>
//#include <mpi.h>
using namespace std;


float data_fudge = 1.0f;

SchlierenRenderer renderer;

void writeArraytoTIFF(uint16_t* image, string filename)
{
    /* 
    This function saves an image array to a TIFF file 
    given by the filename
    */

    TIFF *out = TIFFOpen(filename.c_str(),"w");

    int sampleperpixel = 1;    // 1 for grayscale, 3 for RGB
    int width = renderer._params.width; // image width in pixels
    int height = renderer._params.height; // image height in pixels

    int bits_per_byte = 8;

    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);   // set number of channels per pixel
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bits_per_byte*sizeof(uint16_t));    // set the size of the channels
    TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    tsize_t linebytes = sampleperpixel * width;     // length in memory of one row of pixel in the image.

    uint16_t *buf = NULL;        // buffer used to store the row of pixel information for writing to file

    //    Allocating memory to store the pixels of current row
    if (TIFFScanlineSize(out)==linebytes)
        buf =(uint16_t *)_TIFFmalloc(linebytes);
    else
        buf = (uint16_t *)_TIFFmalloc(TIFFScanlineSize(out));

    // We set the strip size of the file to be size of one row of pixels
    TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width*sampleperpixel));

    //Now writing image to the file one strip at a time
    for (uint32 row = 0; row < height; row++)
    {
        std::memcpy(buf,&image[row*linebytes],linebytes*sizeof(uint16_t));
        if (TIFFWriteScanline(out, buf, row, 0) < 0)
            break;
    }

    (void) TIFFClose(out);

    if (buf)
        _TIFFfree(buf);

}

void display()
{
    /* 
    This function is recursively called by the screen. It calls
    the render() function in schlierenrenderer.cpp to begin
    the image rendering process
    */
    
    cout << "glut display\n";
    static int count = 0;

    std::cout << "On loop number " << count << "\n";

    if (count++ >= 100)
    {
        cout << "finished rendering\n";
        return;
    }
    renderer.render();

    glutPostRedisplay();
    glutSwapBuffers();
}

void keyboard( unsigned char key, int x, int y) {}

double last_x = 0.0;
double last_y = 0.0;
bool dirty_gradient = true;
bool alteringTF = false;
bool rotating = false;

void mouse(int button, int state, int x, int y) 
{
    if (button == GLUT_RIGHT_BUTTON)
    {
        #if USE_TF
        if (state == GLUT_DOWN)
        {
          alteringTF = true;
          dirty_gradient = true;
          last_x = x;
          last_y = y;
        }
        else
        {
          //cudaMemcpy(d_TFBuckets, transferFunction->buckets, sizeof(float4)*transferFunction->numBuckets, cudaMemcpyHostToDevice);
          alteringTF = false;
        }
        #endif
    }
    else if (button == GLUT_LEFT_BUTTON) 
    {
        last_x = x;
        last_y = y;
        if (state == GLUT_DOWN) 
            rotating = true;
        else 
        {
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
    if (alteringTF)
    {
        for(int i =0 ; i < 100; i++) 
        {
            int tx = last_x + (x - last_x)*(i/100.0f);
            int ty = last_y + (y - last_y)*(i/100.0f);
            if (tfRenderable->Contains(tx,window_height - ty))
              tfRenderable->ProcessMouse(tx,window_height -ty);
        }
        clear();
    }
    #endif
    
    float scale = 0.00001;
    
    if (rotating) 
    {
        rot_x += (x-last_x)*scale;
        rot_y += (y-last_y)*scale;

        renderer.rotate(rot_x, rot_y);
        renderer.updateCamera();

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
    vector<string> files, tempFiles, pressFiles;
    DataFile* dataFiles[200];

    files.push_back(string(argv[1]));

    // get filenames
    for(int file = 0; file < files.size(); file++) 
    {
        dataFiles[file] = new DataFile();
        dataFiles[file]->filename = new char[256];
    }
    char** input_files;
    input_files = new char*[files.size()];
    
    // read data from NRRD file
    for( int i = 0; i < files.size(); i++) 
    {
        cout << "file: " << files[i] << endl;
        input_files[i] = new char[files[i].length()];
        strcpy(input_files[i], files[i].c_str());
        strcpy(dataFiles[i]->filename, input_files[i]);
        loadNRRD(dataFiles[i],data_min, data_max);
    }

    // get data parameters
    float* data = dataFiles[0]->data;
    int zrange = dataFiles[0]->sizez;

    // get min and max bound along all three axes
    renderer._params.data_min_bound = make_float3(dataFiles[0]->xmin, dataFiles[0]->ymin, dataFiles[0]->zmin);
    renderer._params.data_max_bound = make_float3(dataFiles[0]->xmax, dataFiles[0]->ymax, dataFiles[0]->zmax);

    // print min and max bound along all three axes
    printf("data_min_bound: %f, %f, %f\n", renderer._params.data_min_bound.x, renderer._params.data_min_bound.y, renderer._params.data_min_bound.z);
    printf("data_max_bound: %f, %f, %f\n", renderer._params.data_max_bound.x, renderer._params.data_max_bound.y, renderer._params.data_max_bound.z);

    //***************************** begin rendering ***********************************
    cout << "setting up renderer\n";
      
    // set cutoff and datascalar to userdefined inputs
    renderer._params.dataScalar = atof(argv[2]);
    renderer._params.cutoffScalar = atof(argv[3]);

    // calculate gradients in the data set
    renderer.setData(data, dataFiles[0]->sizex,dataFiles[0]->sizey,zrange);

    // set step size for ray propagation
    renderer.setStepSize(step_size);

    cout << "setting filter\n";
    renderer.setFilter(new SchlierenPositiveHorizontalKnifeEdgeCutoff());
    // renderer.setFilter(new SchlierenShadowgraphCutoff());
    //renderer.setFilter(new SchlierenImageCutoff(cutoffData));
    //renderer.setFilter(new SchlierenInterforemetryCutoff());

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

    // start rendering mainloop
    printf("Entering Main Loop\n");
    cout << "mainloop\n";
    //glutMainLoop();
  
    cout << "mainloop done\n";

    // iteratively render images without going through glutMainLoop
    for(int i = 1; i<=1; i++)
        renderer.render();

    // get filename to save data
    string out_file(argv[4]);

    // calculate number of pixels in the image
    int num_pixels = renderer._params.width * renderer._params.height;
    int k;

    // initialize image array to be saved as a TIFF file
    uint16_t image_array[num_pixels];
    
    // obtain (truncated) intensity values at each pixel location 
    for(k = 0;k<num_pixels;k++)
        image_array[k] = int(renderer._params.out_rgb[k]); 
    
    // write image_array to TIFF file    
    writeArraytoTIFF(image_array,out_file);

    printf("Finished Writing to file\n");

    // set out_rgb to zero to avoid memory issues
    for(k = 0;k<num_pixels;k++)
    renderer._params.out_rgb[k] = 0;

    // clear variables
    renderer.clear();

    return 0;
}

extern "C"
{
  void loadNRRD(DataFile* datafile, int data_min, int data_max)
  {
    /* This function reads the density data from the NRRD file in addition to 
    variables containing information about the extent of the volume and the grid
    spacing. It also converts the density to refractive index.
    Information about the nrrd file format is available at : 
    http://teem.sourceforge.net/nrrd/lib.html
    */
    
    printf("loading file %s : ", datafile->filename);
    Nrrd* nrrd = nrrdNew();

    // if the file does not exist, print error and exit
    if(nrrdLoad(nrrd, datafile->filename, 0)) 
    {
        char* err=biffGetDone(NRRD);
        cerr << "Failed to open \"" + string(datafile->filename) + "\":  " + string(err) << endl;
        exit(__LINE__);
    }
    
    // obtain number of grid points along each axis 
    int sizex, sizey, sizez;
    sizex = nrrd->axis[0].size;
    sizey = nrrd->axis[1].size;
    sizez = nrrd->axis[2].size;
  
    // get min, max and grid spacing for each axis 
    double xmin, xmax, ymin, ymax, zmin, zmax;
    double del_x, del_y, del_z;

    xmin = nrrd->spaceOrigin[0];
    del_x = nrrd->axis[0].spacing;
    xmax = xmin + sizex*del_x;
 
    ymin = nrrd->spaceOrigin[1];
    del_y = nrrd->axis[1].spacing;
    ymax = ymin + sizey*del_y;
 
    zmin = nrrd->spaceOrigin[2];
    del_z = nrrd->axis[2].spacing;
    zmax = zmin + sizez*del_z;
 
    printf("\n******************** Co-ordinate System Info ******************************\n");
    printf("xmin: %f, xmax: %f, N_x: %d, del_x: %f\n", xmin,xmax,sizex,del_x);
    printf("ymin: %f, ymax: %f, N_y: %d, del_y: %f\n", ymin,ymax,sizey,del_y);
    printf("zmin: %f, zmax: %f, N_z: %d, del_z: %f\n", zmin,zmax,sizez,del_z);
    

    // not sure what these statements do    
    if (data_max > sizez)
      data_max = sizez;
    if (sizez > (data_max-data_min))
      sizez = data_max-data_min;

    printf(" size: %f %f %f ", float(sizex), float(sizey), float(sizez));

    // initialize data array and max and min variables
    float* data = new float[sizex*sizey*sizez];
    float min = FLT_MAX;
    float max = -FLT_MAX;
    float* dataNrrd = (float*)nrrd->data;
    float* datai = data;

    // set GladStone Dale constant (cm^3/g) for refractive index calculation
	float K = 0.226; 
	
    for(int i = 0; i < sizex; i++) 
    {
      for(int j = 0; j < sizey; j++) 
      {
        for( int k = 0; k < sizez; k++) 
        {
            // read density data from file  
			*datai = (*dataNrrd)*data_fudge;
			
			// convert density to refractive index
		  	*datai = 1.0 + K/1000.0*(*datai);
            
            // update max and min values
            if (*datai > max)
                max = *datai;
            if (*datai < min)
                min = *datai;

            datai++;
            dataNrrd++;

        }
      }
    }

    // transfer data to pointer
    datafile->data = data;
    datafile->sizex = sizex;
    datafile->sizey = sizey;
    datafile->sizez = sizez;

    datafile->xmin = xmin;
    datafile->xmax = xmax;
    datafile->ymin = ymin;
    datafile->ymax = ymax;
    datafile->zmin = zmin;
    datafile->zmax = zmax;
    datafile->del_x = del_x;
    datafile->del_y = del_y;
    datafile->del_z = del_z;

    // close file
    nrrdNuke(nrrd);
    printf("  ...done\n");
    printf("Min: %f, Max: %f \n",min, max);
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
