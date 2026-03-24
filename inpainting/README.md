# SIGK - Projekt 1: Modyfikacja Obrazów (Inpainting)

## Architektura Modelu: Zmodyfikowany U-Net

Bazowa architektura to sieć typu U-Net, która została poddana znaczącym modyfikacjom w celu dostosowania jej do specyfiki zadania inpaintingu i generowania ostrych, realistycznych tekstur.

### 1. Czterokanałowe Wejście (4-Channel Input)
Standardowe sieci U-Net przyjmują obrazy 3-kanałowe (RGB). Nasz model na wejściu przyjmuje 4 kanały: 3 kanały zamaskowanego obrazu RGB oraz 1 kanał binarnej maski, która precyzyjnie informuje sieć, gdzie znajdują się brakujące piksele.

### 2. Konwolucje Bramkowane (Gated Convolutions)
Klasyczne konwolucje traktują czarne dziury jako faktyczne dane, co powoduje rozmywanie się "pustki" na zdrowe piksele w warstwach ukrytych i tzw. "zatrucie" połączeń omijających (skip connections). 
Aby temu zapobiec, standardowe bloki konwolucyjne zostały zastąpione przez Gated Convolutions (inspirowane architekturą DeepFill v2). Warstwa ta składa się z dwóch równoległych konwolucji:
* Pierwsza ekstrahuje cechy obrazu.
* Druga (przepuszczona przez funkcję Sigmoid) działa jako dynamiczna maska (uwaga/attention), ucząc się ignorować uszkodzone obszary i skupiać wyłącznie na prawidłowych pikselach.

### 3. Zwiększone Pole Widzenia (Dilated Convolutions)
Aby model był w stanie poprawnie zrekonstruować największe otwory (32x32 piksele), jego najgłębsza część (bottleneck) musi "widzieć" wystarczająco szeroki kontekst. Zastosowano tam konwolucje dylacyjne (Dilated Convolutions), które rozszerzają pole widzenia (receptive field) bez zwiększania liczby parametrów modelu.

### 4. Bezpieczny Upsampling (Bilinear + Conv2d)
Zamiast standardowej warstwy `ConvTranspose2d` w dekoderze (która często generuje artefakty szachownicy), zastosowano dwuetapowy proces:
1.  Bilinearna interpolacja (Upsampling) zwiększająca rozdzielczość.
2.  Standardowa konwolucja wyrównująca cechy.

---

## Funkcja Straty (Loss Function)

Optymalizacja modelu odbywa się poprzez minimalizację złożonej funkcji straty, która wymusza nie tylko poprawność matematyczną, ale również percepcyjną jakość tekstur.

1.  **Weighted L1 Loss:** Podstawowa miara błędu absolutnego, podzielona na obszar prawidłowy (valid) oraz obszar dziury (hole) z odpowiednimi wagami. Dba o ogólną strukturę kolorów.
2.  **Perceptual Loss (VGG):** Porównuje mapy cech (feature maps) wyekstrahowane przez pre-trenowaną sieć VGG16. Zapobiega rozmyciom (blur) i poprawia lokalne struktury oraz krawędzie.
3.  **Gram Matrix Style Loss:** Kluczowy element generujący realistyczne tekstury (np. futro, trawa). Oblicza macierz Grama z cech VGG16, zmuszając model do naśladowania statystyk korelacji cech (stylu/tekstury) obrazu oryginalnego wewnątrz rekonstruowanej dziury.

---

## Ewaluacja i Metryki

Wyniki są porównywane z metodą bazową `INPAINT_TELEA` zaimplementowaną w OpenCV.
Zgodnie z wymaganiami, ewaluacja jakości rozwiązań jest przeprowadzana na całym zbiorze testowym DIV2K z wykorzystaniem następujących metryk:
* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index Measure)
* **LPIPS** (Learned Perceptual Image Patch Similarity)

## Instrukcje Uruchomienia (CLI)

Projekt został zaprojektowany w sposób modułowy. Poniżej znajdują się komendy pozwalające na trenowanie oraz ewaluację modelu. Należy upewnić się, że uruchamiamy skrypty z poziomu głównego katalogu projektu.

### Trenowanie Modelu
Aby rozpocząć proces trenowania z wykorzystaniem pliku konfiguracyjnego (np. dla dziur o rozmiarze 32x32)a:

```bash
python src/train.py --config configs/train_32.yaml
```
Skrypt automatycznie zapisze najlepsze (oraz ostatnie) wagi modelu (na podstawie metryki PSNR na zbiorze walidacyjnym) do katalogu results/checkpoints/.

### Ewaluacja Metody Bazowej (Telea)
Aby wygenerować metryki i wizualizacje dla algorytmu Telea na zbiorze testowym dla maski 32x32:

```bash
python src/evaluate.py --method telea --split test --mask_size 32
```
### Ewaluacja Zbudowanego Modelu (U-Net)
Aby przetestować wytrenowaną sieć U-Net, należy wskazać ścieżkę do pliku konfiguracyjnego oraz punktu kontrolnego (checkpoint) z wagami modelu:

```bash
python src/evaluate.py \
  --config configs/train_32.yaml \
  --method unet \
  --split test \
  --mask_size 32 \
  --checkpoint results/checkpoints/unet_mask32/best.pt
```
Wszystkie zagregowane metryki oraz przykładowe wizualizacje (zestawienia ground truth, maski, obrazu zamaskowanego i predykcji) zostaną zapisane w odpowiednim podkatalogu w folderze results/eval/.