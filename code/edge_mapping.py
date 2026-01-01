#importing necessary libraries
import os
import cv2
import numpy as np

# Preprocessing
def preprocess(gray):
    """
    Contrast normalization + edge-preserving smoothing
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.bilateralFilter(enhanced, 9, 75, 75)

# Cost Map Construction
def build_cost(gray, prefer_bottom=True):
    """
    Build a robust cost map using:
    - Vertical gradient magnitude
    - Local vertical coherence (noise suppression)
    - Positional bias (top / bottom)
    """
    h, w = gray.shape

    # Vertical gradient
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gy)

    # Normalize
    mag /= (mag.max() + 1e-6)

    # Coherence: suppress isolated noise
    coherence = cv2.GaussianBlur(mag, (1, 9), 0)
    coherence /= (coherence.max() + 1e-6)

    # Strong coherent edges → low cost
    cost = 1.0 - (0.7 * mag + 0.3 * coherence)

    # Positional bias
    y = np.linspace(0, 1, h).reshape(-1, 1)
    if prefer_bottom:
        cost += 0.5 * (1.0 - y)
    else:
        cost += 0.5 * y

    return cost

# Robust Band Estimation
def estimate_band(cost, region="bottom", ratio=0.35):
    """
    Estimate search band using percentile statistics (robust to noise)
    """
    h, _ = cost.shape

    if region == "bottom":
        start = int(h * (1.0 - ratio))
        end = h
    else:
        start = 0
        end = int(h * ratio)

    row_score = np.percentile(cost[start:end], 10, axis=1)
    center = start + np.argmin(row_score)

    band_half = int(h * 0.08)
    top = max(0, center - band_half)
    bottom = min(h - 1, center + band_half)

    return top, bottom

# Dynamic Programming Path
def find_path(cost, band_top, band_bottom):
    """
    Column-wise DP with adaptive smoothness
    """
    h, w = cost.shape

    dp = np.full((h, w), np.inf, np.float32)
    back = np.zeros((h, w), np.int32)

    dp[band_top:band_bottom, 0] = cost[band_top:band_bottom, 0]

    smooth_lambda = 0.2 * np.mean(cost[band_top:band_bottom])

    for x in range(1, w):
        for y in range(band_top, band_bottom):
            y0 = max(band_top, y - 2)
            y1 = min(band_bottom - 1, y + 2)

            prev = dp[y0:y1 + 1, x - 1]
            idx = np.argmin(prev)

            smooth_penalty = smooth_lambda * abs((y0 + idx) - y)
            dp[y, x] = cost[y, x] + prev[idx] + smooth_penalty
            back[y, x] = y0 + idx

    y = band_top + np.argmin(dp[band_top:band_bottom, -1])

    path = []
    for x in reversed(range(w)):
        path.append((x, y))
        y = back[y, x]

    return np.array(path[::-1])

# Contour Detection
def detect_bottom_contour(gray):
    cost = build_cost(gray, prefer_bottom=True)
    band_top, band_bottom = estimate_band(cost, "bottom")
    return find_path(cost, band_top, band_bottom)

def detect_top_contour(gray):
    cost = build_cost(gray, prefer_bottom=False)
    band_top, band_bottom = estimate_band(cost, "top")
    return find_path(cost, band_top, band_bottom)

# Mask & Selective Denoising
def build_mask(top, bottom, shape):
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for x in range(w):
        y1, y2 = sorted([top[x, 1], bottom[x, 1]])
        mask[y1:y2 + 1, x] = 255
    return mask

def denoise(gray, mask):
    den = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    out = gray.copy()
    out[mask == 255] = den[mask == 255]
    return out

# Visualization
def draw(image, path, color=(0, 0, 255)):
    out = image.copy()
    for i in range(1, len(path)):
        cv2.line(out, tuple(path[i - 1]), tuple(path[i]), color, 2)
    return out

# Main Pipeline
def process_image(path, out_dir):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Cannot read {path}")

    name = os.path.splitext(os.path.basename(path))[0]

    pre = preprocess(gray)

    bottom = detect_bottom_contour(pre)
    top = detect_top_contour(pre)

    mask = build_mask(top, bottom, gray.shape)
    denoised = denoise(gray, mask)

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = draw(color, bottom)

    cv2.imwrite(os.path.join(out_dir, f"{name}_overlay.png"), overlay)
    cv2.imwrite(os.path.join(out_dir, f"{name}_denoised.png"), denoised)

# Batch Runner
if __name__ == "__main__":
    INPUT_DIR = "../inputs"
    OUTPUT_DIR = "../outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith(".bmp"):
            process_image(os.path.join(INPUT_DIR, f), OUTPUT_DIR)

    print("Processing complete.")