import cv2
import numpy as np
from skimage.filters import frangi
from skimage.measure import regionprops, label

img_path = r"original_009245722fa4.png"
img = cv2.imread(img_path)
img = cv2.resize(img, (384, 384))

# 1. Landmarks
imgH, imgW = img.shape[:2]
r_chan = img[:,:,2] 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel)
circles = cv2.HoughCircles(r_closed, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=20, maxRadius=45)
od_mask = np.zeros((imgH, imgW), dtype=np.uint8)
od_center = (imgW//2, imgH//2)
od_radius = 30
if circles is not None:
    circles = np.uint16(np.around(circles))
    od_center = (circles[0, 0][0], circles[0, 0][1])
    od_radius = circles[0, 0][2]
else:
    blurred = cv2.GaussianBlur(r_closed, (21, 21), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    od_center = max_loc
cv2.circle(od_mask, od_center, od_radius, 255, -1)

# 2. Vessels
g_chan = img[:,:,1]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
g_enh = clahe.apply(g_chan)
g_inv = cv2.bitwise_not(g_enh)
vesselness = frangi(g_inv, sigmas=range(1, 8, 2), black_ridges=False)
vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Lower the threshold to pick up MORE vessels (to fix broken vessels)
thresh = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
vessel_mask = vesselness_norm > (thresh * 0.3)  # WAS 0.5
vessel_mask = vessel_mask.astype(np.uint8)

# Dilate the vessel mask to cover the "halos" and broken fragments!
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
vessel_mask_dilated = cv2.dilate(vessel_mask, kernel_dilate)

# 3. MA
kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# Don't use CLAHE for MA either! It creates noise. Use raw g_chan.
ma_candidates = cv2.morphologyEx(g_chan, cv2.MORPH_BLACKHAT, kernel_ma)

# Try a stricter threshold
ma_mask = ma_candidates > 15 # WAS 15

ma_mask[vessel_mask_dilated > 0] = False
ma_mask[od_mask > 0] = False

# Region props
labeled_ma = label(ma_mask.astype(np.uint8))
regions = regionprops(labeled_ma)
clean_ma_mask = np.zeros_like(ma_mask, dtype=np.uint8)
ma_count = 0
for props in regions:
    # MAs must be perfectly round (eccentricity < 0.7) and small (area < 20)
    if 2 < props.area <= 20 and props.eccentricity < 0.8:
        coords = props.coords
        clean_ma_mask[coords[:, 0], coords[:, 1]] = 1
        ma_count += 1

print(f"Original MA threshold: 15, no vessel dilation -> Resulting MAs: {ma_count}")

# Output to disk to verify
cv2.imwrite(r"original_009245722fa4_vessel_fixed.png", (vessel_mask_dilated * 255).astype(np.uint8))

