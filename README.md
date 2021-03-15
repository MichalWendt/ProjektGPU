# Zbiór Mandelbrota na GPU

Autorstwa Michała Wendt

## Wstęp historyczny

Benoît B. Mandelbrot był francusko-amerykańskim matematykiem polsko-żydowskiego pochodzenia.
Urodził się 20 listopada 1924 roku w Warszawie. Zmarł 14 października 2010 roku w Cambridge
(Massachusetts). Praktykował wiele działów matematyki. Zainspirował się pracami francuskich
matematyków: Gastona Julii i Pierre’a Fatou. Poprzez badanie iteracji funkcji zespolonych za pomocą
nowoczesnych maszyn dostępnych dzięki pracy w IBM stworzył wykresy nazwane fraktalami. Dzisiaj
jest przede wszystkim rozpoznawany jako „ojciec geometrii fraktalnej”. Jest to spowodowane
odkryciem i opisaniem „zbioru Mandelbrota” oraz wymyślenia samego słowa „fraktal”.

## Czym jest Fraktal?

Samo pojęcie pojawiło się w świecie matematyki w latach 70 XX wieku.

Fraktal to pewien obiekt, który w sensie geometrycznym jest samopodobny, czyli jakaś jego część jest
podobna do całości. Spotykamy się także z określeniem jako nieskończenie złożony. Czyli taki, który
możemy teoretycznie nieskończenie powiększać.

Matematycy uogólniając opisują fraktale jako spełniające większość lub wszystkie z poniższych
warunków:

```
> ma nietrywialną strukturę w każdej skali,
> struktura ta nie daje się łatwo opisać w języku tradycyjnej geometrii euklidesowej,
> jest samopodobny, jeśli nie w sensie dokładnym, to przybliżonym lub stochastycznym
> jego wymiar Hausdorffa jest większy niż jego wymiar topologiczny
> ma względnie prostą definicję rekurencyjną,
> ma naturalny („poszarpany”, „kłębiasty” itp.) wygląd.
```
Ważnym fraktalem jakim będziemy zajmowali się w tej pracy jest brzeg Zbioru Mandelbrota. Jest to
specyficzny fraktal który istnieje tylko w świecie matematycznym. Sam brzeg zbioru Mandelbrota, nie
spełnia wszystkich tych warunków tzn. ma wymiar Hausdorffa równy 2, czyli taki sam jak jego wymiar
topologiczny. Sam Zbiór Mandelbrota jest podzbiorem płaszczyzny zespolonej i nie jest fraktalem,
ponieważ nie jest samopodobny, co zostało dowiedzione przez chińską matematyczkę Tan Lei.

## Wzór na Zbiór Mandelbrota

Nasz zbiór możemy opisać jako te punkty p(zespolone) dla których ciąg(Zi)infi=0 zdefiniowany
równaniem rekurencyjnym

nie dąży do nieskończoności

Można wykazać, że jest to równoważne z


Podsumowując jednym zdaniem:

## Omówienie kodu programu

Używany przez mnie wzór będzie wydawał się o wiele łatwiejszy do zrozumienia niż ta definicja. Od
tego wzoru właśnie chciałbym zacząć tę pracę.

![obraz](https://user-images.githubusercontent.com/58120860/111216823-2c23bd00-85d5-11eb-900a-822c94ac2d8e.png)

Ogólnie mówiąc, aby określić czy punkt należy do zbioru (tutaj czy chcemy go pokolorować) musimy
sprawdzić czy wszystkie z wybranych wyrazów początkowych ciągu zn spełniają warunek |Zn| < 2 (W
samym programie jestem zmuszony używać 4 przez niedokładność wartości zmiennoprzecinkowych).
Inaczej mówiąc wartość podciągu nie zaczyna drastycznie rosnąć. Takim wartością ustawiamy
konkretny kolor, a punktom pobliskim kolor podobny i wymieszany z innymi pobliskimi. Jeżeli nasz
iterator przekroczył ustaloną maksymalną wartość iteracji punkt będzie zamalowany na czarno.

Opiszmy co dzieje się na zdjęciu.

Na początku musimy zadeklarować nasze zmienne x oraz y równe 0. Tutaj tworze także zmienną
pomocniczą.

Użyjemy także iteratora ‘i’ obliczającego ilość iteracji potrzebnych do wcześniej wspomnianego
„drastycznego wzrostu”. Wnętrze pętli while jest dosłownym zaimplementowaniem wzoru
wspomnianego wyżej. Jeżeli którykolwiek z warunków (koniec iteracji lub drastyczny wzrost ciągu)
nastąpi możemy zakończyć pętle i zwrócić naszą ilość iteracji.

Kolejnymi ważnymi do omówienia funkcjami są „MandelbrotOn***”

Są to dwie funkcje. Jedna w pliku .cu druga w .cpp. Funkcje są bardzo podobne ale z drobnymi
różnicami, więc postanowiłem je omówić razem.

Ważna częścią funkcji działającej na GPU jest poruszanie się po siatce bloków. Przypisuje ona
kolejnym watkom część obrazu, którą muszą się zająć

![obraz](https://user-images.githubusercontent.com/58120860/111216837-31810780-85d5-11eb-9e97-4ce7d3038ead.png)

W dalszej części musimy oczywiście przejść po całym obrazie. W przypadku CPU są to zwykłe dwie
pętle przechodzące pierw wszerz, a następnie po kolei w pionie do kolejnych poziomów.

![obraz](https://user-images.githubusercontent.com/58120860/111216859-3940ac00-85d5-11eb-8cba-cd26a5315d1e.png)

W funkcji GPU jest to tylko pojedyncza kontrola sprawdzająca czy nie wyszliśmy poza obszar
malowany, ponieważ bloki wątków odpowiadają kolejnym pikselom.

Później w obu wersjach w identyczny sposób wywołujemy funkcje liczącą ilość iteracji (pokazaną
wcześniej) oraz przemnażamy barwy RGB przez wynikową liczbę.

![obraz](https://user-images.githubusercontent.com/58120860/111216869-3c3b9c80-85d5-11eb-8465-aad9a00aa967.png)

Teraz możemy przypisać kolor do konkretnego punktu naszego obrazu.

![obraz](https://user-images.githubusercontent.com/58120860/111216878-3f368d00-85d5-11eb-900c-3c1522df640d.png)

Ostatnią z funkcji związanych z Mandelbrotem będzie funkcja renderująca „render”. Jej głównym
przeznaczeniem jest obliczanie nowego obszaru wyświetlanego i wywołanie odpowiedniej funkcji
MandelbrotOn***.

Musimy oczywiście zacząć od deklaracji zmiennych, określenia miejsca zasobów dla CUDY oraz
obliczenie nowej wyświetlanej lokalizacji.

![obraz](https://user-images.githubusercontent.com/58120860/111216891-42317d80-85d5-11eb-9e28-acfd4ca2929c.png)

Gdy dane są już gotowe możemy wywołać funkcje zależnie od ustawienia urządzenia na CPU lub
GPU. Trzeba pamiętać, że w przypadku CPU otrzymane dane trzeba wysłać do GPU w celu ich
wyświetlenia na ekranie. Na koniec sprzątamy po sobie już niepotrzebne zasoby CUDY.

![obraz](https://user-images.githubusercontent.com/58120860/111216917-465d9b00-85d5-11eb-8d68-ae242c8eec97.png)

Powyższe polecenia są wywoływane przez display, która odpowiada za wyświetlenie nam obrazu w
naszej ramce, tworzenie ramki pojawi się za chwile. Do wyświetlania obrazu używam OpenGl

(dokładny opis technologii pozostawiam na koniec). Teraz tylko wspomnę, że jest to biblioteka
umożliwiająca tworzenie grafiki. OpenGl musi oczywiście wiedzieć trochę o tym co chcemy
wyświetlić, więc informujemy go w której zmiennej powinien szukać potrzebnych tekstur, o
wymiarowości obrazu oraz typie shaderów.

![obraz](https://user-images.githubusercontent.com/58120860/111216946-49f12200-85d5-11eb-9a8e-e23d03cb9471.png)

Tutaj musimy utworzyć prostokąt o 4 wierzchołkach w którym będziemy mogli wyświetlić obraz. Gdy
wszystkie informacje są gotowe, zamykamy program GL-a i podmieniamy otrzymany wynik z
poprzednio wyświetlanym.

![obraz](https://user-images.githubusercontent.com/58120860/111216985-54abb700-85d5-11eb-987c-81c27cd171aa.png)

Kolejną funkcję omówię krótko. Jest to prosty switch do wykonywania konkretnych poleceń przy
wejściu z klawiatury. Nasłuch na wejście pojawi się później. Możemy tu zmienić skale, przesuwać się,
edytować zmienne lub zakończyć działanie programu.

![obraz](https://user-images.githubusercontent.com/58120860/111217008-5b3a2e80-85d5-11eb-8ac8-fde77c4d8dc2.png)

TimeEvent to krótka funkcja odświeżająca nasze okno w określonych w REFRESH_DELAY odstępach
czasu.

![obraz](https://user-images.githubusercontent.com/58120860/111217018-5e351f00-85d5-11eb-986f-c71789fea076.png)

Tutaj już w dużym stopniu gotowa funkcja shadera OpenGL generująca program z danego codu i
sprawdzająca ewentualne błędy zaistniałe podczas tego procesu.

![obraz](https://user-images.githubusercontent.com/58120860/111217034-60977900-85d5-11eb-8fbc-5687e482fd14.png)

W funkcji initOpenGLBuffers jak sama nazwa mówi stworzymy miejsce na dane OpenGL-a. W tym
celu musimy usunąć ewentualne już istniejące zapisy. I zaalokować miejsce na nowy obraz.

![obraz](https://user-images.githubusercontent.com/58120860/111217044-642b0000-85d5-11eb-9d85-2fe7cc30622d.png)

Teraz za pomocą funkcji GL-a możemy wygenerować wszystkie potrzebne nam dane w celu
generowania tekstur.

![obraz](https://user-images.githubusercontent.com/58120860/111217051-65f4c380-85d5-11eb-943f-8ef8a4676846.png)

Edycja okna wyświetlanego może odbywać się na dwa sposoby, użytkownik może zmienić domyślną
rozdzielczość obrazu na początku pliku .cpp lub podczas jego trwania przytrzymać przycisk myszy na
ramce obrazu i rozciągnąć lub skurczyć w dowolnie wybrany sposób. Obraz wyświetlany będzie
dostosowywał się natychmiastowo.

![obraz](https://user-images.githubusercontent.com/58120860/111217058-68571d80-85d5-11eb-9fa7-fb728f549e1d.png)

Zostało nam jeszcze kilka ważnych linii kodu w funkcji main. Pierwszy fragment odpowiada za
inicjalizacje danych potrzebnych dla poprawnego działania CUDY. Czyli informacje o urządzeniach
zdolnych do wykonywania wymaganych obliczeń.

![obraz](https://user-images.githubusercontent.com/58120860/111217067-6beaa480-85d5-11eb-91f8-5d2d7872085e.png)

Drugi zestaw to początkowo zbierane dane dla GL-a (wraz z aktywacją funkcji), takie jak: w jakim
formacie i jakiej rozdzielczości ma działać, gdzie ma zacząć, że ma nasłuchiwać klawiaturę czy jak
często ma odświeżać swój proces.

![obraz](https://user-images.githubusercontent.com/58120860/111217073-6e4cfe80-85d5-11eb-941e-c90d683d0c5f.png)

Jeszcze sprawdzamy czy użytkownik ma dostatecznie nową wersję GL ( dodałem już bibliotekę więc
nie powinno być z tym problemu) i informuje o ewentualnym błędzie.

I przechodzimy do funkcji teksturującej już wcześniej wytłumaczonej.

![obraz](https://user-images.githubusercontent.com/58120860/111217080-70af5880-85d5-11eb-955f-ececc1588280.png)

Na koniec informujemy OpenGL, żeby zapętlił swój program w celu nasłuchu na dalsze polecenia.

## Opis użytych technologii:

Program został napisany przy użyciu technologii Nvidia CUDA w języku C++. Korzysta z gotowych
bibliotek CUDY oraz OpenGL-a. Powstał w środowisku VisualStudio.

Podstawowym aspektem przemawiającym za przewagą GPU nad CPU w tym typie problemu jest fakt
wyspecjalizowania do intensywnych obliczeniowo, wysoce równoległych obliczeń bycia
zaprojektowanym w taki sposób, by więcej tranzystorów było przeznaczonych do przetwarzania
danych, a nie do buforowania danych i kontroli przepływu.

**Visual Studio** – środowisko programistyczne od firmy Microsoft

Wystarczy pobrać ze strony producenta:

**https://visualstudio.microsoft.com/pl/free-developer-offers/**

I przejść przez prosty proces instalacji pamiętając o wybraniu opcji „Desktop development with C++”
w szczegółach instalacji.


**CUDA** – technologia stworzona i rozwijana przez firmę Nvidia. Jej głównym założeniem jest użycie
procesorów wielordzeniowych (najczęściej kart graficznych) w celu użycia mocy obliczeniowej w
sposób wydajniejszy niż tradycyjny tj. sekwencyjny. Wykonywanie wielu obliczeń w tym samym
czasie znacznie przyspieszyło obliczenia problemów numerycznych. Tym samym umożliwiając
wykonywanie bardzo skomplikowanych działań ogromną ilość razy w bardzo krótkim czasie.

Cuda zawiera w swoim środowisku programistycznym własny kompilator (nvcc), debugger (cuda-
gdb), profiler, interfejs programowania aplikacji oraz wiele przydatnych bibliotek, które można
wykorzystać w językach Python, Fortran, Java, C# oraz Matlab.

Z Cudą możemy spotkać się na wielu systemach operacyjnych ponieważ aktualnie wspiera zarówno
Windowsa, Linuxa jak i Mac OS X.

**Kilka zalet pokazujących, że warto używać cudy:**

```
 Model pamięci procesora ściśle odpowiadający architekturze sprzętowej, co umożliwia
świadome, efektywne wykorzystywanie dostępnych zasobów GPU, w tym pamięci
współdzielonej. Pamięć ta jest współdzielona przez wszystkie wątki w tzw. bloku (zwykle 128-
512 wątków). Można jej używać jako programowalnej pamięci typu cache.
 Pełna kompatybilność wsteczna programów.
 Dostępność na wszystkich kartach firmy NVIDIA począwszy od serii GeForce 8
```
**Aby zacząć używać cudy musimy upewnić się, że nasza karta graficzna jest do tego zdolna. W tym
celu sprawdzamy czy znajduje się na liście GPU na stronie Nvidii:
https://developer.nvidia.com/cuda-gpus**

**Jeżeli nasza karta graficzna jest na liście możemy pobrać zestaw narzędzi CUDY:**

**https://developer.nvidia.com/cuda-downloads**

Proces instalacji jest dość prosty, wystarczy wybrać ścieżkę instalacji i typ instalacji.

**OpenGL** (Open Graphics Library) –uniwersalne i open-sourcowe API służące do renderowania grafiki
wektorowej 2D i 3D. Jego głównym przeznaczeniem jest interakcja z GPU w celu uzyskania
renderowania przyspieszanego sprzętowo.

API jest zdefiniowane jako zestaw funkcji, które mogą być wywoływane przez program klienta, wraz z
zestawem nazwanych stałych całkowitych.

Instalacja opiera się na pobraniu odpowiednich paczek do projektu. W moim projekcie te paczki są
już zaimplementowane.
