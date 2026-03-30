# Projekt SIGK — Inpainting obrazów

Implementacja zadania **2.4 Inpainting** z wykorzystaniem **PyTorch** i zbioru **DIV2K**. Celem projektu jest odtwarzanie losowo wyciętych fragmentów obrazu o rozmiarze **3×3** oraz **32×32** i porównanie własnego modelu z metodą bazową **OpenCV `inpaint(..., INPAINT_TELEA)`**.

## Założenia zadania

Zgodnie z opisem projektu:
- używany jest zbiór **DIV2K**,
- obrazy do trenowania i walidacji mają rozdzielczość **256×256**,
- w zadaniu inpainting wycinane są losowe obszary **3×3** oraz **32×32**,
- metodą bazową jest **OpenCV Telea**,
- ocena jakości odbywa się metrykami **PSNR**, **SSIM** oraz **LPIPS** liczonymi na całym zbiorze testowym. 

## Opis architektury końcowej

Końcowy model jest modyfikacją lekkiego **U-Neta** dla zadania inpaintingu.

### Wejście i wyjście

- **Wejście**: tensor o kształcie **[B, 4, H, W]**
  - 3 kanały obrazu z zamaskowanym fragmentem,
  - 1 kanał binarnej maski.
- **Wyjście**: tensor **[B, 3, H, W]** z przewidywaną zawartością RGB.

### Główne elementy architektury

1. **Encoder–decoder typu U-Net**
   - cztery poziomy enkodera i dekodera,
   - połączenia skip-connection między odpowiadającymi sobie poziomami,
   - liczba kanałów bazowych sterowana parametrem `base_channels`.

2. **Gated Convolution**
   - podstawowy blok konwolucyjny używa warstwy `GatedConv2d`,
   - oprócz klasycznej konwolucji obliczana jest dodatkowa maska aktywacji,
   - wynik bloku ma postać: `feature * sigmoid(mask)`.

3. **Upsampling bilinearny w dekoderze**
   - zamiast transposed convolution zastosowano:
     - `Upsample(scale_factor=2, mode="bilinear")`,
     - następnie zwykłą konwolucję 3×3,
   - zmniejsza to ryzyko artefaktów typu checkerboard.

4. **Bottleneck z konwolucjami dylatowanymi**
   - w najgłębszej części sieci użyto dwóch bloków `DilatedConvBlock`,
   - rozszerza to pole recepcyjne bez dalszego zmniejszania rozdzielczości,
   - pomaga wykorzystywać szerszy kontekst przy wypełnianiu brakującego obszaru.

5. **Aktywacja wyjściowa `Sigmoid`**
   - wyjście modelu jest ograniczane do zakresu **[0, 1]**, zgodnego z normalizacją obrazów.

### Strata użyta w treningu

W treningu zastosowano złożoną funkcję straty:

- **Weighted L1 loss**:
  - większy nacisk na obszar brakujący (`hole region`),
  - mniejszy nacisk na obszar znany (`valid region`),
- **VGG perceptual loss**:
  - różnica cech wyznaczanych przez kolejne fragmenty sieci **VGG16**,
  - liczona na obrazie złożonym z przewidywania w dziurze i oryginału poza dziurą.

Całkowita strata ma postać:

```text
L_total = L_weighted_L1 + λ * L_perceptual
```

W praktyce taki wariant dawał lepsze wyniki niż najprostszy U-Net uczony wyłącznie przez L1, chociaż w przypadku maski 32×32 nadal nie przebił metody Telea.

## Pipeline treningu i ewaluacji

### Dane

- trening: `DIV2K_train_LR_bicubic`
- test: `DIV2K_valid_LR_bicubic`
- obrazy są przycinane / przygotowywane do rozmiaru **256×256**
- maska binarna jest dołączana jako dodatkowy kanał wejściowy

### Trening

- optymalizator: **Adam**
- najlepszy checkpoint wybierany na podstawie **val PSNR**
- zapisywane są:
  - `best.pt`
  - `last.pt`
  - `history.json`
  - wykres `training_curves.png`

### Ewaluacja

Skrypt `src/evaluate.py` obsługuje dwa tryby:
- `telea` — metoda bazowa OpenCV,
- `unet` — ewaluacja wytrenowanego modelu.

Dla każdego obrazu zapisywane są również przykłady jakościowe:
- obraz oryginalny,
- obraz zamaskowany,
- predykcja,
- figura porównawcza.

## Wyniki

Opis projektu wymaga raportowania metryk **PSNR / SSIM / LPIPS**, a dodatkowo rozszerzono ewaluację o metrykę **SNE (Squared Norm Error)**.

Wyniki raportowane są w dwóch wariantach:
- **full** – dla całego obrazu,
- **hole** – tylko w obszarze maski (najbardziej miarodajne dla inpaintingu).

---

## Inpainting 32×32 — zbiór testowy DIV2K

### Metryki globalne (cały obraz)

| Metoda | PSNR ↑ | SSIM ↑ | LPIPS ↓ | SNE ↓ |
|---|---:|---:|---:|---:|
| `opencv telea inpaint 32x32` | 38.0547 | 0.9908 | 0.0120 | 61.7336 |
| `gated-dilated unet 32x32` | 34.3963 | 0.9875 | 0.0155 | 111.6530 |

---

### Metryki lokalne (tylko maska — właściwy inpainting)

| Metoda | PSNR ↑ | SSIM ↑ | LPIPS ↓ | SNE ↓ |
|---|---:|---:|---:|---:|
| `opencv telea inpaint 32x32` | 19.9929 | 0.3854 | 0.2088 | 61.7336 |
| `gated-dilated unet 32x32` | 16.3345 | 0.2984 | 0.1820 | 111.6530 |

---

## Krótki komentarz do wyników

- Metoda klasyczna **Telea** przewyższa model U-Net zarówno w metrykach globalnych, jak i lokalnych.
- Różnica jest szczególnie widoczna w metrykach liczonych **w obrębie maski**, które najlepiej odzwierciedlają jakość inpaintingu.
- Model U-Net generuje bardziej rozmyte rekonstrukcje, co skutkuje niższym PSNR i SSIM oraz wyższym błędem SNE.
- Jednocześnie LPIPS dla U-Net jest niższy niż dla Telea w obszarze maski, co sugeruje, że rekonstrukcje są perceptualnie bardziej „gładkie”, choć mniej dokładne pikselowo.

## Wyniki dla wariantu 3×3

| Metoda | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| `opencv telea inpaint 3x3` | 68.0839 | 1.0000 | 0.0000 |
| `gated-dilated unet 3x3` | 65.7399 | 0.9999 | 0.0001 |

### Uwaga

Dla maski **3×3** nie raportowano metryk lokalnych (hole), ponieważ:
- SSIM wymaga większego kontekstu przestrzennego,
- bardzo mały rozmiar maski powoduje niestabilność i brak interpretowalności wyników.

## Najważniejsze pliki

- `src/models/blocks.py` — definicje `GatedConv2d`, bloków enkodera/dekodera i bloków dylatowanych
- `src/models/unet.py` — końcowa architektura U-Net
- `src/losses/losses.py` — strata L1 + perceptual loss
- `src/train.py` — trening modelu
- `src/evaluate.py` — ewaluacja modelu i baseline'u

## Najważniejsze komendy

### 1. Smoke test środowiska

```bash
python scripts/smoke_test.py
```

### 2. Wygenerowanie splitów danych

```bash
python scripts/create_splits.py
```

### 3. Podgląd próbek z datasetu

Maska 32×32:

```bash
python scripts/preview_dataset.py --split train --mask_size 32 --num_samples 4
```

Maska 3×3:

```bash
python scripts/preview_dataset.py --split train --mask_size 3 --num_samples 4
```

### 4. Smoke test modelu

```bash
python scripts/test_model.py
```

### 5. Trening modelu

Wariant 32×32:

```bash
python src/train.py --config configs/train_32.yaml --max_val_metric_samples 10
```

Wariant 3×3:

```bash
python src/train.py --config configs/train_3.yaml --max_val_metric_samples 10
```

### 6. Ewaluacja baseline'u Telea

Maska 32×32:

```bash
python src/evaluate.py --method telea --split test --mask_size 32
```

Maska 3×3:

```bash
python src/evaluate.py --method telea --split test --mask_size 3
```

### 7. Ewaluacja wytrenowanego modelu U-Net

Maska 32×32:

```bash
python src/evaluate.py \
  --config configs/train_32.yaml \
  --method unet \
  --split test \
  --mask_size 32 \
  --checkpoint results/checkpoints/unet_mask32/best.pt
```

Maska 3×3:

```bash
python src/evaluate.py \
  --config configs/train_3.yaml \
  --method unet \
  --split test \
  --mask_size 3 \
  --checkpoint results/checkpoints/unet_mask3/best.pt
```