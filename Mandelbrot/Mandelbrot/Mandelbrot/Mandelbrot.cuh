#ifndef _MANDELBROT_KERNEL_cuh_   //plik naglowkowy dla mandelbrot_cuda.cu
#define _MANDELBROT_KERNEL_cuh_

#define REFRESH_DELAY 10        // czestotliwosc odswierzania (w milisekundach)
#define MINI(a,b) ((a < b) ? a : b)     // wlasne minimum
#define BLOCKDIM_X 32           // wymiary bloku watkow (mozna edytowac w celu przyspieszenia obliczen)
#define BLOCKDIM_Y 32   

#include <helper_gl.h>      // biblioteki OpenGL
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>   // CUDA runtime
#include <helper_cuda.h>    // CudacheckErrors sa zawarte w tej bibliotece
#include <complex>
#include <chrono>
#include <omp.h>

using namespace std;

extern "C" void RunMandelbrotOnGPU(uchar4 * GPUImage, const int imageWidth, const int imageHeight, const int MaxIter, const double XStart, const double YStart,
    const double Scale, const uchar4 colors, const int frame, const int numSMs, bool CalcTime);	//deklaracja funckji rzutujacej generator pikseli na floaty

template<class T>
__device__ inline int MandelOblicz(const double PozX, const double PozY, const int MaxIter)   // Funkcja obliczajaca ilosc iteracji dla kazdego piksela
{
    double x = 0, y = 0, xtemp;  // zmienne wzoru Mandelbrota
    int i = 0;              // licznik iteracji

    while (i < MaxIter && (x * x + y * y < 4)) // iterujemy do wybranej wartosci sprawdzajac czy nasze wartosci nie zaczynaja drastycznie rosnac    
    {   //(sprawdzamy czy modu³ elementu ci¹gu jest mniejszy ni¿ 2)
        xtemp = x * x - y * y + PozX;   // pomocnicza przetrzymuje nowa wartosc x na czas obliczania y
        y = 2 * x * y + PozY;   // nowa wartosc y
        x = xtemp;  // nowa wartosc x
        i = i + 1;  // iterowanie
    }
    return i >= MaxIter ? 0 : i;     // sprawdzamy czy nie otrzymalismy wartosci wiekszych niz nasz ogranicznik
}
#endif