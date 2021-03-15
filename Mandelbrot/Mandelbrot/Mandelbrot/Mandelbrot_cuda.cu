#include "Mandelbrot.cuh"

template<class T>
__global__ void MandelbrotOnGPU(uchar4* GPUImage, const int imageWidth, const int imageHeight, const int MaxIter, const double XStart, const double YStart,
    const double Scale, const uchar4 colors, const int klatka, const int gridWidth, const int numBlocks)  // funckja kolorujaca piksele dla GPU
{
    for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks; blockIndex += gridDim.x)  // zapetlamy po wszystkich blokach
    {
        unsigned int BlockDimX = blockIndex % gridWidth;   // obliczamy wielkosc bloku w siatce
        unsigned int BlockDimY = blockIndex / gridWidth;

        const int i = blockDim.y * BlockDimY + threadIdx.y;// okreslenie bloku do przetwarzania
        const int j = blockDim.x * BlockDimX + threadIdx.x;

        if ((j < imageWidth) && (i < imageHeight))      // sprawdzamy czy nie opuscilismy obszaru wyswietlanego
        {
            const double PozX = (double)j * Scale + XStart;       // obliczamy nasza lokalizacje
            const double PozY = (double)i * Scale + YStart;

            int iter = MandelOblicz<T>(PozX, PozY, MaxIter);    // obliczamy ilosc iteracji dla kazdego piksela

            uchar4 color;
            color.x = iter * colors.x;      // ustawienie RGB piksela w zaleznosci od zmiennej iter
            color.y = iter * colors.y;
            color.z = iter * colors.z;

            int pixel = imageWidth * i + j; // okreslamy nasza pozycje w przestrzeni jednowymiarowej

            int klatka1 = klatka + 1;
            int klatka2 = klatka1 / 2;
            GPUImage[pixel].x = (GPUImage[pixel].x * klatka + color.x + klatka2) / klatka1;    // ustawianie kolejnych RGB dla pikseli 
            GPUImage[pixel].y = (GPUImage[pixel].y * klatka + color.y + klatka2) / klatka1;
            GPUImage[pixel].z = (GPUImage[pixel].z * klatka + color.z + klatka2) / klatka1;
        }
    }
}

inline int CzyPodzielne(int a, int b) // zwiekszenie pola siatki o 1 w przypadku podania wartosci rozmiarow niepodzielnych przez wymiary bloku watkow
{
    if ((a % b) != 0)
        return (a / b + 1);
    else
        return (a / b);
}

void RunMandelbrotOnGPU(uchar4* GPUImage, const int imageWidth, const int imageHeight, const int MaxIter, const double XStart, const double YStart,
    const double Scale, const uchar4 colors, const int klatka, const int ProcNum, bool CalcTime) // wywolanie funkcji mandelbora z rzutowaniem na float i ilosc watkow
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(CzyPodzielne(imageWidth, BLOCKDIM_X), CzyPodzielne(imageHeight, BLOCKDIM_Y)); // sprawdzanie podzielnosci bloku watkow na wymiary okna i tworzenie z nich siatki

    int numWorkerBlocks = ProcNum;

    if (CalcTime)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        MandelbrotOnGPU<double> << <numWorkerBlocks, threads >> > (GPUImage, imageWidth, imageHeight, MaxIter, XStart, YStart, Scale, colors, klatka, grid.x, grid.x * grid.y);   // wersja na podwojnej precyzji (dluzej liczy)

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        printf("Obliczenia na GPU zajely: %f ms\n", elapsed_ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    else
    {
        //MandelbrotOnGPU<float> << <numWorkerBlocks, threads >> > (GPUImage, imageWidth, imageHeight, MaxIter, (float)XStart, (float)YStart, (float)Scale, colors, klatka, grid.x, grid.x * grid.y); // wersja na floatach da nam mniejsza mozliwosc przyblizania
        MandelbrotOnGPU<double> << <numWorkerBlocks, threads >> > (GPUImage, imageWidth, imageHeight, MaxIter, XStart, YStart, Scale, colors, klatka, grid.x, grid.x * grid.y);   // wersja na podwojnej precyzji (dluzej liczy)
    }
}