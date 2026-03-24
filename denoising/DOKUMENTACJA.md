# Odszumianie obrazów (Denoising)

## 1. Opis zadania

Celem zadania było odszumianie obrazów z gaussowskim szumem o współczynnikach σ = 0.01 i σ = 0.03. Obrazy są znormalizowane do przedziału [0.0, 1.0]. Jako metodę bazową zastosowano filtr bilateralny (`denoise_bilateral` z biblioteki skimage), a nasze rozwiązanie to sieć neuronowa w architekturze U-Net.

---

## 2. Przygotowanie danych

Do trenowania i ewaluacji użyliśmy zbioru **DIV2K**:
- **Trening:** `DIV2K_train_LR_bicubic` (90% trening, 10% walidacja)
- **Test:** `DIV2K_valid_LR_bicubic` (100 obrazów)

Obrazy są przycinane do rozdzielczości **256×256 pikseli**.

Szum gaussowski jest dodawany przy użyciu funkcji `random_noise` z biblioteki **skimage**.



---

## 3. Architektura modelu — U-Net

Zaproponowaliśmy klasyczną architekturę **U-Net**, składającą się z enkodera, wąskiego gardła (bottleneck) i dekodera z połączeniami pomijającymi (skip connections).

### Bloki składowe

- **ConvBlock** — dwa bloki `Conv2d(3×3) → BatchNorm → ReLU` (bez zmiany rozdzielczości)
- **DownBlock** — `MaxPool2d(2×2)` + `ConvBlock` (zmniejsza rozdzielczość dwukrotnie)
- **UpBlock** — `ConvTranspose2d(2×2)` (zwiększa rozdzielczość dwukrotnie) + konkatenacja ze skip connection + `ConvBlock`

### Struktura sieci

| Warstwa       | Typ        | Kanały wyjściowe |
|---------------|------------|-----------------|
| enc1          | ConvBlock  | 64              |
| enc2          | DownBlock  | 128             |
| enc3          | DownBlock  | 256             |
| enc4          | DownBlock  | 512             |
| bottleneck    | DownBlock  | 1024            |
| dec1          | UpBlock    | 512             |
| dec2          | UpBlock    | 256             |
| dec3          | UpBlock    | 128             |
| dec4          | UpBlock    | 64              |
| out_conv      | Conv2d(1×1)| 3               |
| out_activation| Sigmoid    | —               |

Na wyjściu sieć generuje obraz RGB o tych samych wymiarach co wejście. Funkcja aktywacji Sigmoid zapewnia wartości w przedziale [0.0, 1.0].

---

## 4. Trening

Trenowaliśmy osobny model dla każdego poziomu szumu (σ = 0.01 i σ = 0.03).

| Parametr       | Wartość |
|----------------|---------|
| Optymalizator  | Adam    |
| Learning rate  | 0.001   |
| Funkcja straty | L1      |
| Batch size     | 8       |
| Epoki          | 30      |

---

## 5. Metoda bazowa — filtr bilateralny

Jako metodę bazową zastosowaliśmy **filtr bilateralny** (`denoise_bilateral` z skimage) z domyślnymi parametrami biblioteki (`sigma_color` wyznaczane automatycznie jako 10% zakresu obrazu, `sigma_spatial=1`). Filtr ten wygładza obraz, jednocześnie starając się zachować krawędzie dzięki uwzględnieniu zarówno odległości przestrzennej, jak i podobieństwa wartości pikseli.

---

## 6. Wyniki

Ewaluacja została przeprowadzona na zbiorze testowym DIV2K (100 obrazów). Użyte metryki:
- **SNE** — stosunek sygnału do błędu (wyższy = lepiej)
- **PSNR** — szczytowy stosunek sygnał/szum w dB (wyższy = lepiej)
- **SSIM** — strukturalne podobieństwo obrazów, zakres [0, 1] (wyższy = lepiej)
- **LPIPS** — percepcyjna miara podobieństwa (niższy = lepiej)

### Szum σ = 0.01

| Metoda                        | SNE   | PSNR (dB) | SSIM   | LPIPS  |
|-------------------------------|-------|-----------|--------|--------|
| bilateral_gaussian_noise_001  | 14.20 | 20.79     | 0.5340 | 0.2505 |
| unet_gaussian_noise_001       | **25.98** | **32.56** | **0.9556** | **0.0333** |

### Szum σ = 0.03

| Metoda                        | SNE   | PSNR (dB) | SSIM   | LPIPS  |
|-------------------------------|-------|-----------|--------|--------|
| bilateral_gaussian_noise_003  | 14.15 | 20.73     | 0.5192 | 0.2615 |
| unet_gaussian_noise_003       | **24.28** | **30.86** | **0.9294** | **0.0485** |

---



## 7. Wnioski

U-Net przewyższa filtr bilateralny we wszystkich metrykach i dla obu poziomów szumu. Różnica w PSNR wynosi ponad **10 dB**, a w SSIM sieć uzyskuje wartości powyżej 0.93 w porównaniu do około 0.52 dla filtra. Wyniki potwierdzają, że podejście oparte na sieci neuronowej jest zdecydowanie skuteczniejsze w zadaniu odszumiania obrazów niż klasyczne metody filtracji.

Dla wyższego poziomu szumu (σ = 0.03) wyniki U-Netu są nieco gorsze niż dla σ = 0.01, co jest spodziewane — silniejszy szum jest trudniejszy do usunięcia. Mimo to sieć radzi sobie znacznie lepiej niż metoda bazowa.


