# Edge Mapping Assignment

## Overview
This project implements a **classical computer vision solution** for robust edge (contour) mapping and selective denoising, as specified in the assignment. The goal is to:

1. **Accurately map the bottom contour** across the full width of an image, even in the presence of noise, low contrast, high spatial frequency variation, and discontinuities.
2. **Accurately map the top contour** corresponding to the same layered structure.
3. **Denoise only the region between the detected top and bottom contours**, leaving the rest of the image unchanged.

The solution is implemented entirely in Python using traditional image processing techniques (no deep learning), and is designed to be robust across all example image types provided in the assignment.

---

## Folder Structure

```
project/
│
├── code/
│   └── edge_mapping.py   # Main implementation
│
├── inputs/
│   └── *.bmp             # Input images (provided)
│
├── outputs/
│   ├── *_overlay.png     # Bottom contour overlay visualization
│   └── *_denoised.png    # Selectively denoised output
│
└── readme.md             # This file
```

---

## Key Design Principles

- **Classical CV only**: No neural networks or learning-based methods.
- **Global optimization**: Contours are found using dynamic programming instead of local thresholding or edge linking.
- **Robustness first**: Noise, discontinuities, and weak edges are handled explicitly.
- **Exact region control**: Denoising is applied strictly between true top and bottom contours.

---

## Algorithm Pipeline (End-to-End)

### 1. Preprocessing

Each input image is converted to grayscale and preprocessed to stabilize intensity and suppress noise while preserving edges:

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** improves visibility in low-intensity images.
- **Bilateral filtering** smooths noise while preserving edge structure.

This step ensures that subsequent gradient-based processing is consistent across all image types.

---

### 2. Cost Map Construction

Instead of directly thresholding edges, the problem is formulated as a **cost minimization task**.

For each pixel, a cost value is computed based on:

- **Vertical gradient magnitude** (Sobel Y): highlights horizontal band edges.
- **Vertical coherence**: suppresses isolated noisy responses by encouraging vertically consistent edges.
- **Positional bias**:
  - Bottom contour detection prefers lower rows.
  - Top contour detection prefers upper rows.

Strong, coherent edges at plausible vertical positions receive **low cost**, while noise and irrelevant structures receive **high cost**.

---

### 3. Robust Search Band Estimation

To avoid being distracted by noise elsewhere in the image:

- A **search band** is estimated separately for the top and bottom contours.
- Row-wise cost statistics are computed using **percentiles** (instead of means) for robustness.
- The dynamic programming search is restricted to this narrow vertical band.

This greatly improves stability in noisy and discontinuous images.

---

### 4. Dynamic Programming Contour Detection

Contour detection is performed using **column-wise dynamic programming**, similar to seam carving:

- Exactly one pixel is selected per column.
- A **smoothness penalty** discourages abrupt vertical jumps.
- The smoothness weight is adaptive, based on average band cost.

This approach ensures:
- Full-width continuity
- Automatic bridging of gaps and discontinuities
- Resistance to local noise near the edge

The same algorithm is used independently for:
- **Bottom contour detection**
- **Top contour detection**

---

### 5. Mask Construction

Once both contours are detected:

- A binary mask is created column-by-column.
- Pixels strictly between the top and bottom contours are marked as valid.

This mask precisely defines the region that should be denoised.

---

### 6. Selective Denoising

Denoising is applied **only inside the masked region**:

- **Non-Local Means denoising** is used for effective noise suppression.
- Pixels outside the mask remain completely unchanged.

This directly satisfies the requirement that noise near the edge or outside the contour band should not affect results.

---

### 7. Visualization and Output

For each input image, two outputs are generated:

1. **Overlay image** (`*_overlay.png`)
   - Displays the detected bottom contour as a red line on the original image.

2. **Denoised image** (`*_denoised.png`)
   - Shows the selectively denoised region between contours.

All outputs are saved in the `outputs/` directory.

---

## How to Run

1. Place all provided `.bmp` images in the `inputs/` directory.
2. From the `code/` directory, run:

```
python edge_mapping.py
```

3. Results will be written to the `outputs/` directory.

---

## Notes

- The algorithm is deterministic and does not require parameter tuning per image.
- All parameters were chosen to balance robustness and simplicity.
- The same pipeline works across clean, noisy, low-intensity, and discontinuous images as demonstrated by the outputs.

---
