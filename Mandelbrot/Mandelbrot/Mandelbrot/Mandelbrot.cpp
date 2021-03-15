#include "Mandelbrot.cuh"    // moj plik naglowkowy

GLuint BufforName, TextureName, ShaderName;     // zmienne OpenGL na typie UNSIGNED_INT
struct cudaGraphicsResource* PboReso;           // zasob mapowania CUDY
static const char* shader_code = "!!ARBfp1.0\n" "TEX result.color, fragment.texcoord, texture[0], 2D; \n" "END";   // Shader OpenGL-a do wyswietlania tekstur typu float

uchar4* HostImage = 0;      // obraz po stronie CPU ("naszej")
uchar4* GPUImage = NULL;    // obraz po stronie GPU
uchar4 Colors;

int imageWidth = 1280, imageHeight = 720; // Zmienne okreslajace poczatkowa wielkosc okna  (wybralem rozdzielczosc 720p)
int MaxIter = 512;              // Okreslony limit iteracji dla piksela (szegolowosc)
int Klatka = 0;                 // wygladzanie renderowania
int MultiProcNumber = 0;        // liczba multiprocesorow
int GLVersion = 1;              // werjsa OpenGL (na potrzeby zabezpieczenia)

double XStart = -0.5;   // okreslenie punktu startowego wyswietlanego obrazu(lekko przesuniety poniewaz tak wyglada lepiej)
double YStart = 0.0;
double Scale = 3.5;     // skala obrazu (przy takiej widac caly zbior)

bool CzyCPU = false;    // zmienna okreslajaca czy uzywamy CPU czy GPU
bool CzyOMP = true;     // zmienna okreslajaca czy uzywamy rownoleglych watkow
bool CalcTimeGPU = false;  // zmienna okreslajaca czy obliczyc czas pracy na GPU
bool CalcTimeCPU = false;  // zmienna okreslajaca czy obliczyc czas pracy na CPU

//definicje funckji (opisy funckji sa przy samych funkcjach)
void MandelbrotOncpu(uchar4*, const int, const int, const int, const double, const double, const double, const uchar4, const int);
void render();
void display(void);
void ButtonInput(unsigned char, int, int);
void timerEvent(int);
GLuint compileASMShader(GLenum, const char*);
void initOpenGLBuffers(int, int);
void ResizeWindow(int,int);

int main(int argc, char** argv) // program glowny
{
    printf("Witamy w generatorze Mandelbrota\n");    // wiadomosc powitalna

    XStart = -0.5;    // ustaweienie zmiennych na domyslne
    YStart = 0.0;
    Scale = 3.5;
    MaxIter = 512;
    Klatka = 0;

    Colors.x = 5;   // ustawienie kolorow
    Colors.y = 6;
    Colors.z = 7;

    int dev = 0;    // inicjalizowanie danych CUDY
    dev = findCudaDevice(argc, (const char **)argv);            // wyszukiwanie sprzetu obliczajacego w podwojnej precyzji

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev)); // pobieramy informacje o znalezionym urzadzeniu 
    GLVersion = deviceProp.major * 10 + deviceProp.minor;

    MultiProcNumber = deviceProp.multiProcessorCount;

    glutInit(&argc, argv);   // inicjalizowanie funkcji OpenGl-a 

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);   // ustalenie trybu wyswietlania
    glutInitWindowSize(imageWidth, imageHeight);    // ustaweienie paramatrow wielkosci
    glutInitWindowPosition(0, 0);                   // ustawienie pozycji okna na ekranie (od lewego gornego rogu)
    glutCreateWindow("Mandelbrot w CUDA");          // nazwanie okna

    glutDisplayFunc(display);                       // zaladowanie funkcji wyswietlenia 
    glutKeyboardFunc(ButtonInput);                  // zaladowanie funkcji wejsc klawiatury 
    glutReshapeFunc(ResizeWindow);                  // zaladowanie funkcji zmiany wielkosci okna 
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);    // zaladowanie funkcji odswiezania

    if (!isGLVersionSupported(1, 5) || !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))  // wylapywanie bledu przestarzalej wersji OpenGL
    {
        fprintf(stderr, "Blad: Wersja OpenGL jest starsza niz 1.5");    // informacja w razie bledu
        fflush(stderr);
        return false;
    }

    initOpenGLBuffers(imageWidth, imageHeight); // uruchomienie okna o podanych wymiarach

    printf("\n");
    printf("Nacisnij 'S' aby zmienic tryb renderowania (CPU lub GPU)\n");  // wypisanie opcji wejscia z klawiatury
    printf("Nacisnij 'R' aby przywrocic domyslne wartosci zmiennych\n");
    printf("Nacisnij 'G' aby zwiekszyc szegolowosc\n");
    printf("Nacisnij 'L' aby zmniejszyc szegolowosc\n");
    printf("Nacisnij 'M' aby zmienic wieloprocesorowosc\n");
    printf("Nacisnij 'C' aby obliczyc czas wykonania w aktulnym trybie\n");
    printf("Nacisnij 'E' aby zakonczyc dzialanie\n");
    printf("Uzyj strzalek klawiatury numerycznej aby sie poruszac\n");
    printf("\n");

    glutMainLoop(); //wchodzimy do petli przetwarzania zdarzen GLUT
}

// funckja kolorujaca piksele dla CPU
void MandelbrotOncpu(uchar4* CPUImage, const int imageWidth, const int imageHeight, const int MaxIter, const double XStart, const double YStart, const double Scale, const uchar4 Colors, const int klatka)
{
    if (CzyOMP) // wersja wieloprocesorowa
    {
        auto start_time = std::chrono::high_resolution_clock::now();    // rozpoczecie obliczania czasu
        #pragma omp parallel for    // zrownoleglenie petli for
        for (int i = 0; i < imageHeight; i++)    // przechodzimy po calej wysokosci
        {
            for (int j = 0; j < imageWidth; j++) // i szerokosci obrazu
            {
                const double PozX = (double)j * Scale + XStart;      // obliczamy nasza aktualna pozycje na obrazie
                const double PozY = (double)i * Scale + YStart;

                int iter = MandelOblicz<int>(PozX, PozY, MaxIter);    // obliczamy ilosc iteracji dla kazdego piksela

                uchar4 color;
                color.x = iter * Colors.x;      // ustawienie RGB piksela w zaleznosci od zmiennej iter
                color.y = iter * Colors.y;
                color.z = iter * Colors.z;

                int pixel = imageWidth * i + j; // okreslamy nasza pozycje w przestrzeni jednowymiarowej

                int klatka1 = klatka + 1;
                int klatka2 = klatka1 / 2;
                CPUImage[pixel].x = (CPUImage[pixel].x * klatka + color.x + klatka2) / klatka1; // ustawianie kolejnych RGB dla pikseli na naszym obrazie
                CPUImage[pixel].y = (CPUImage[pixel].y * klatka + color.y + klatka2) / klatka1;
                CPUImage[pixel].z = (CPUImage[pixel].z * klatka + color.z + klatka2) / klatka1;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();  // zakonczenie obliczania czasu
        if (CalcTimeCPU)    // wypisanie czasu obliczen jezeli uzytkownik o to poprosil
        {
            std::chrono::duration<double> diff = end_time - start_time;
            printf("Obliczenia na CPU wieloprocesorowym zajely: %f ms\n", std::chrono::duration <double, milli >(diff).count());
        }
    }
    else    // wersja jednoprocesorowa
    {
        auto start_time = std::chrono::high_resolution_clock::now();    // rozpoczecie obliczania czasu
        for (int i = 0; i < imageHeight; i++)    // przechodzimy po calej wysokosci
        {
            for (int j = 0; j < imageWidth; j++) // i szerokosci obrazu
            {
                const double PozX = (double)j * Scale + XStart;      // obliczamy nasza aktualna pozycje na obrazie
                const double PozY = (double)i * Scale + YStart;

                int iter = MandelOblicz<int>(PozX, PozY, MaxIter);    // obliczamy ilosc iteracji dla kazdego piksela

                uchar4 color;
                color.x = iter * Colors.x;      // ustawienie RGB piksela w zaleznosci od zmiennej iter
                color.y = iter * Colors.y;
                color.z = iter * Colors.z;

                int pixel = imageWidth * i + j; // okreslamy nasza pozycje w przestrzeni jednowymiarowej

                int klatka1 = klatka + 1;
                int klatka2 = klatka1 / 2;
                CPUImage[pixel].x = (CPUImage[pixel].x * klatka + color.x + klatka2) / klatka1; // ustawianie kolejnych RGB dla pikseli na naszym obrazie
                CPUImage[pixel].y = (CPUImage[pixel].y * klatka + color.y + klatka2) / klatka1;
                CPUImage[pixel].z = (CPUImage[pixel].z * klatka + color.z + klatka2) / klatka1;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();  // zakonczenie obliczania czasu
        if (CalcTimeCPU)    // wypisanie czasu obliczen jezeli uzytkownik o to poprosil
        {
            std::chrono::duration<double> diff = end_time - start_time;
            printf("Obliczenia na CPU jednoprocesorowym zajely: %f ms\n", std::chrono::duration <double, milli >(diff).count());
        }
    }
}

void render()  // renderowanie obrazu zbioru dla CPU lub GPU przy uzyciu CUDY
{
    size_t num_bytes;                                           // wielkosc wskaznika
    checkCudaErrors(cudaGraphicsMapResources(1, &PboReso, 0));  // mapowanie zasobow dla CUDY
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&GPUImage, &num_bytes, PboReso));    // uzyskanie wskaŸnika przez który mo¿na uzyskaæ dostêp do zmapowanego zasobu

    static const unsigned char pairData[8][2] = { {34,54}, {0,0}, {6,63}, {83,11},{96,32}, {57,23}, {36,86}, {30,31} };

    float xpom, ypom;   // wartosci aktualnego przesuniecia
    xpom = (1.0f / 128.0f) * (0.5f + (float)pairData[Klatka % 8][0]);
    ypom = (1.0f / 128.0f) * (0.5f + (float)pairData[Klatka % 8][1]);

    double s = Scale / (float)imageWidth;      // obliczanie nowej skali
    double x = (xpom - (double)imageWidth * 0.5f) * s + XStart; // obliczanie nowego przesuniecia
    double y = (ypom - (double)imageHeight * 0.5f) * s + YStart;
    if (CzyCPU) //renderowanie przy uzyciu CPU
    {
        MandelbrotOncpu(HostImage, imageWidth, imageHeight, MaxIter, x, y, s, Colors, Klatka++); // wywolanie funkcji kolorujacej piksele
        checkCudaErrors(cudaMemcpy(GPUImage, HostImage, imageWidth * imageHeight * sizeof(uchar4), cudaMemcpyHostToDevice));        // wyslanie obliczonych danych do wyswietlenia
        CalcTimeCPU = false;
    }
    else        //renderowanie przy uzyciu GPU
    {
        RunMandelbrotOnGPU(GPUImage, imageWidth, imageHeight, MaxIter, x, y, s, Colors, Klatka++, MultiProcNumber, CalcTimeGPU);    // wywolanie funkcji kolorujacej piksele
        CalcTimeGPU = false;
    }
    checkCudaErrors(cudaGraphicsUnmapResources(1, &PboReso, 0));  // Usuwamy mapowanie zasobow graficznych CUDY
}

void timerEvent(int value)      // funckja odswierzajaca
{
    if (glutGetWindow())        // uzyskujemy identyfikator aktualnego okna (gdyby okna nie bylo to otrzymamy blad i nie bedziemy odswierzac)
    {
        glutPostRedisplay();    // odswiezenie okna (zaznaczamy je jako wymagajace ponownego wyswietlenia)
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);    // rejestrujemy wywolanie zwrotne zegara, które ma zostaæ wyzwolone w okreslonej liczbie milisekund
    }
}

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;                  // zmienna id typu UNSIGNED_INT
    glGenProgramsARB(1, &program_id);   // wygenerowanie jednej nazwy programu
    glBindProgramARB(program_type, program_id); // uzycie programu
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);    // aktualizacja naszego programu obiektem "code"

    GLint ErrorNum;         // zmiennan numeru bledu
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &ErrorNum);   // zwracamy wartosc pozycji bledu

    if (ErrorNum != -1)     // jezeli pozycja jest "prawidlowa" (znaczy faktycznie mamy blad)
    {
        fprintf(stderr, "Blad programu na pozycji: %d\n%s\n", (int)ErrorNum, glGetString(GL_PROGRAM_ERROR_STRING_ARB));   // wypisanie bledu pobranego z numeru programu
        return 0;
    }
    return program_id;      // zwrocenie id programu shadera
}

void initOpenGLBuffers(int NewWidth, int NewHight)    // tworzenie tekstur dla opengla
{
    free(HostImage);    // jezeli istnieje jakis obraz po naszej stronie zwalniamy zajmowana przez niego pamiec
    HostImage = 0;      // i ustawiamy falsz jako brak obrazu
    glDeleteTextures(1, &TextureName);  // usuwamy jedna podana stara teksture
    TextureName = 0;                    // i ustawiamy falsz jako brak tekstury
    cudaGraphicsUnregisterResource(PboReso);  // wyrejestrowujemy zasob graficzny CUDY
    glDeleteBuffers(1, &BufforName);    // usuwamy jeden podany bufor
    BufforName = 0;                     // i ustawiamy falsz jako brak buforu

    HostImage = (uchar4 *)malloc(NewWidth * NewHight * 4);    // alokowanie nowego bufora na obraz

    glEnable(GL_TEXTURE_2D);        // wlaczenie gluta  //tworzenie tekstu GL
    glGenTextures(1, &TextureName); // generowanie tekstur
    glBindTexture(GL_TEXTURE_2D, TextureName);   //przypisanie tekstur do zmiennej
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_LINEAR);       // ten parametr tekstur pozwala na pobranie sredniej z 4 sasiadujacych z nasza pozycja
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // okresla, ze tekstura powinna zostac zmniejszona przy zmianie szczegolowosci
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, NewWidth, NewHight, 0, GL_RGBA, GL_UNSIGNED_BYTE, HostImage);    // uscislamy ze chcemy uzywac przestrzeni 2D
  
    glGenBuffers(1, &BufforName);   // tworzenie PBO, PBO to obiekt buforu dla pikseli
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, BufforName);   //przypisywanie bufora do PBO
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, NewWidth * NewHight * 4, HostImage, GL_STREAM_COPY); // tworzymy i inicjujemy miejsce na dane bufora

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&PboReso, BufforName, cudaGraphicsMapFlagsWriteDiscard));    // zarejestrujemy zasob graficzny CUDY

    ShaderName = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code); // zaladowanie programu shaderow
}

void ResizeWindow(int NewWidth, int NewHight)  // funkcja przeksztalcajaca okno (kiedy zmieniamy jego wielkosc)
{
    glMatrixMode(GL_MODELVIEW);     // okreslenie ktorej macierzy uzywamy
    glLoadIdentity();               // podmiana macierzy biezacej z macierza tozsamosci

    glMatrixMode(GL_PROJECTION);    // okreslenie ktorej macierzy uzywamy
    glLoadIdentity();               // podmiana macierzy biezacej z macierza tozsamosci
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);  // mnozenie macierzy  // zwieksza obraz z 1/4 do calosci

    glViewport(0, 0, NewWidth, NewHight); // ustalenie poczatkowych zmiennych
}

void display(void)  // funkcja wyswietlania OpenGL
{
    render();      // wywolanie funkcji renderujacej obraz Mandelbrota

    glBindTexture(GL_TEXTURE_2D, TextureName);   // przypisanie tekstur "TextureName" do GL_TEXTURE_2D
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, (char*)0);    // uscislamy ze chcemy tworzyc tekstury dwuwymiarowe

    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, ShaderName);   // urzycie programu o nazwie "ShaderName" w  GL_FRAGMENT_PROGRAM_ARB
    glEnable(GL_FRAGMENT_PROGRAM_ARB);  // wlaczenie shaderow  GL po stronie serwera
    glDisable(GL_DEPTH_TEST);           // wylaczenie testow glebi w GL

    glBegin(GL_QUADS);  // ustawiamy ograniczenie do 4 wierzcholkow
    glTexCoord2f(0, 0); // ustawiamy koordynaty tekstur
    glVertex2f(0, 0);   // zapisujemy te koordynaty jako wierzcholek
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();    // konczymy ustawianie

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);    // usuniecie przypisania naszych tekstur
    glDisable(GL_FRAGMENT_PROGRAM_ARB); // zakoczenie programu

    glutSwapBuffers();  // zamieniamy buffory z aktualnie uzywanym
}

void ButtonInput(unsigned char i, int, int)     // funkcje wejscia z klawiatury przy uzyciu OpenGL
{
    switch (i)
    {
    case '8':       // przesuniecie w gore  // chcialem uzyc GLUT_KEY_UP ale zamiast strzalek otrzymuje przypisanie np. "d"
        YStart += 0.10f * Scale;
        Klatka = 0;
        break;
    case '2':       // przesuniecie w dol
        YStart -= 0.10f * Scale;
        Klatka = 0;
        break;
    case '6':       // przesuniecie w prawo
        XStart += 0.10f * Scale;
        Klatka = 0;
        break;
    case '4':       // przesuniecie w lewo
        XStart -= 0.10f * Scale;
        Klatka = 0;
        break;

    case '+':       // przyblizenie
        Scale /= 1.5f;
        Klatka = 0;
        break;
    case '-':       // oddalenie
        Scale *= 1.5f;
        Klatka = 0;
        break;

    case 'r':       // Przywrocenie wszystkich zmiennych do domyslnych wartosci (R od Reset)
    case 'R':
        printf("Wartosci zmiennych zostaly zresetowane.\n");
        CzyCPU = false;
        CzyOMP = true;
        XStart = -0.5;
        YStart = 0.0;
        Scale = 3.5;
        Colors.x = 5;
        Colors.y = 6;
        Colors.z = 7;
        MaxIter = 512;
        Klatka = 0;
        break;

    case 'G':       //zwiekszanie dokladnosci   (G od Greater)
    case 'g':
        if (2 * MaxIter <= MINI(MultiProcNumber * (GLVersion < 20 ? 512 : 2048), 0x4000))
        {
            MaxIter *= 2;
            Klatka = 0;
        }
        printf("Szczegolowosc: %d\n", MaxIter);
        break;
    case 'L':       //zmniejszanie dokladnosci  (L od Lower)
    case 'l':
        if (MaxIter > 2)
        {
            MaxIter /= 2;
            Klatka = 0;
        }
        printf("Szczegolowosc: %d\n", MaxIter);
        break;

    case 's':      //zmiana sposobu renderowania CPU lub GPU   (S od Switch)
    case 'S':
        if (CzyCPU) // jezeli bylismy w trybie CPU 
        {
            CzyCPU = false;  // przechodzimy na tryb GPU
            printf("Uzywasz wersji GPU\n");
        }
        else
        {
            CzyCPU = true;   // w przeciwnym wypadu z GPU na CPU
            CzyOMP = true;   // i zmieniamy tryb na wieloprocesorowy
            printf("Uzywasz wersji CPU\n");
            printf("Uzywasz trybu wieloprocesorowego\n");
        }
        Klatka = 0;
        glutDestroyMenu(glutGetMenu());     // zmiana trybu w oknie gluta
        break;

    case 'm':   // zmiana trybu cpu na jeden lub wiele procesorow
    case 'M':
        if (CzyOMP) // jezeli bylismy w trybie wieloprocesorowym
        {
            CzyOMP = false;  // przechodzimy na tryb pojedynczy
            printf("Uzywasz trybu jednoprocesorowego\n");
        }
        else
        {
            CzyOMP = true;   // w przeciwnym wypadu na wieloprocesorowy
            printf("Uzywasz trybu wieloprocesorowego\n");
        }
        break;

    case 'c':   // zmiana obliczania czasu pracy CPU lub GPU
    case 'C':
        if (CzyCPU) // jezeli bylismy w trybie CPU 
        {
            CalcTimeCPU = true;
        }
        else
        {
            CalcTimeGPU = true;
        }
        break;

    case 'e':   //zakonczenie procesu (E od Exit)
    case 'E':
        printf("Dziekuje za uzycie generatora Mandelbrota\n");
        glutDestroyWindow(glutGetWindow());
        return;
        break;

    default:
        break;
    }
}