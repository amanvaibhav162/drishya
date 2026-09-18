"""
DRISHYA Pipeline Stage 2 — Core Module
=======================================
Single source of truth for all retinal feature extraction functions.

Resolution: 512x512 (default), all structuring elements parameterized.
Incorporates fixes for:
  Bug 1: Hemorrhage — adaptive threshold + green/red hemoglobin attenuation
  Bug 2: Vessels — no fill_holes, reduced closing, thinning + width validation
  Bug 3: OD — adaptive radius via radial intensity profiling
  Bug 4: MA — skeleton-based proximity, tighter thresholds
  Bug 5: Exudates — unified Otsu with proper OD suppression
  + Expanded feature vector (ETDRS quadrant counts, fovea dist, tortuosity)

Authors: DRISHYA Team
"""

import os
import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import frangi, apply_hysteresis_threshold
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize


# ============================================================================
# CONFIGURATION — all sizes scale with resolution
# ============================================================================
DEFAULT_RES = 512  # Target resolution (512x512)

def _scale(px_at_384, res=DEFAULT_RES):
    """Scale a pixel value calibrated at 384px to the target resolution."""
    return max(1, int(round(px_at_384 * res / 384)))


# ============================================================================
# MODULE 1: LANDMARK EXTRACTION (Optic Disc + Fovea)
# ============================================================================

def extract_landmarks(img, res=DEFAULT_RES):
    """
    Locates the Optic Disc (OD) and Fovea.

    Returns:
        od_mask (bool): Binary mask of the optic disc
        od_center (tuple): (x, y) coordinates of OD center
        od_radius (int): Estimated OD radius in pixels
        fovea_loc (tuple): (x, y) coordinates of the fovea
        fovea_mask (bool): Binary mask of fovea exclusion zone
    """
    imgH, imgW = img.shape[:2]

    # --- 1. Optic Disc Detection ---
    # Red channel (index 2 in BGR) isolates OD best — vessels are dark in red
    r_chan = img[:, :, 2]

    # Large morphological closing erases vessels
    close_r = _scale(15, res)
    if close_r % 2 == 0:
        close_r += 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1))
    r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel_close)

    # Heavy Gaussian blur forces max to align with the true disc center
    blur_k = _scale(60, res)
    blur_k = blur_k * 2 + 1
    blurred = cv2.GaussianBlur(r_closed.astype(np.float64), (blur_k, blur_k), 0)

    # Mask out non-retinal pixels
    retina_mask = r_chan > 20
    blurred[~retina_mask] = 0

    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    od_center = max_loc  # (x, y)

    # --- Adaptive Radius Estimation via Radial Intensity Profiling ---
    od_radius = _estimate_od_radius(r_closed, od_center, retina_mask, res)

    # Create OD mask
    od_mask = np.zeros((imgH, imgW), dtype=np.uint8)
    cv2.circle(od_mask, od_center, od_radius, 255, -1)
    od_mask = od_mask > 0

    # --- 2. Fovea Localization ---
    fovea_loc, fovea_mask = _locate_fovea(img, od_center, od_radius, retina_mask, res)

    return od_mask, od_center, od_radius, fovea_loc, fovea_mask


def _estimate_od_radius(r_closed, od_center, retina_mask, res=DEFAULT_RES):
    """
    Estimate OD radius using radial intensity profiling.
    Samples mean intensity at concentric rings from the OD center.
    The OD boundary is where intensity drops sharply.
    """
    imgH, imgW = r_closed.shape[:2]
    cx, cy = od_center

    r_min = _scale(15, res)
    r_max = _scale(55, res)

    radii = np.arange(r_min, r_max + 1)
    mean_intensities = []

    for r in radii:
        angles = np.linspace(0, 2 * np.pi, max(20, r * 4), endpoint=False)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)

        valid = (xs >= 0) & (xs < imgW) & (ys >= 0) & (ys < imgH)
        xs, ys = xs[valid], ys[valid]

        if len(xs) == 0:
            mean_intensities.append(0)
            continue

        vals = r_closed[ys, xs].astype(float)
        retina_vals = vals[retina_mask[ys, xs]]
        mean_intensities.append(np.mean(retina_vals) if len(retina_vals) > 0 else 0)

    mean_intensities = np.array(mean_intensities)

    if len(mean_intensities) < 3:
        return _scale(30, res)

    gradient = np.gradient(mean_intensities)
    min_grad_idx = np.argmin(gradient)
    estimated_radius = int(radii[min_grad_idx])

    estimated_radius = max(_scale(20, res), min(_scale(55, res), estimated_radius))

    return estimated_radius


def _locate_fovea(img, od_center, od_radius, retina_mask, res=DEFAULT_RES):
    """
    Fovea is the darkest region ~2.5 OD diameters from OD center.
    """
    imgH, imgW = img.shape[:2]
    od_x, od_y = od_center
    od_diameter = od_radius * 2

    if od_x < imgW / 2:
        fovea_search_x = od_x + int(2.5 * od_diameter)
    else:
        fovea_search_x = od_x - int(2.5 * od_diameter)

    fovea_search_y = od_y
    search_radius = int(1.0 * od_diameter)

    x_min = max(0, fovea_search_x - search_radius)
    x_max = min(imgW, fovea_search_x + search_radius)
    y_min = max(0, fovea_search_y - search_radius)
    y_max = min(imgH, fovea_search_y + search_radius)

    g_chan = img[:, :, 1]
    g_blur = cv2.GaussianBlur(g_chan, (11, 11), 0)
    roi = g_blur[y_min:y_max, x_min:x_max]

    fovea_mask = np.zeros((imgH, imgW), dtype=bool)

    if roi.size > 0:
        _, _, min_loc, _ = cv2.minMaxLoc(roi)
        fovea_loc = (x_min + min_loc[0], y_min + min_loc[1])
    else:
        fovea_loc = (imgW // 2, imgH // 2)

    # Fovea exclusion zone: ~60px at 384 -> ~80 at 512
    fovea_excl_r = _scale(60, res)
    fovea_mask_uint = np.zeros((imgH, imgW), dtype=np.uint8)
    cv2.circle(fovea_mask_uint, fovea_loc, fovea_excl_r, 255, -1)
    fovea_mask = fovea_mask_uint > 0

    return fovea_loc, fovea_mask


# ============================================================================
# MODULE 2: VESSEL SEGMENTATION
# ============================================================================

def extract_vessels(img, od_mask=None, res=DEFAULT_RES):
    """
    Extracts the vascular network using Frangi vesselness filter with
    hysteresis thresholding for continuous vessel segmentation.

    Returns:
        visual_vessel_mask (bool): Clean vessel mask for display
        suppression_vessel_mask (bool): Dilated vessel mask for aggressive lesion suppression
        vessel_skeleton (bool): 1-pixel-wide skeleton of the vascular tree
    """
    imgH, imgW = img.shape[:2]

    g_chan = img[:, :, 1]

    # Slightly higher CLAHE clipLimit for better vessel-background contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)

    # Finer sigma scale sampling captures both thin capillaries and thick arcades
    sigmas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, float(_scale(8, res))]
    vesselness = frangi(g_inv, sigmas=sigmas, black_ridges=False)
    vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    thresh = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    # --- HYSTERESIS THRESHOLDING for continuous vessels ---
    # High threshold (0.45 * Otsu): seeds strong vessel cores
    # Low threshold (0.18 * Otsu): follows weak capillary segments connected to cores
    # This preserves continuity through bifurcations and thin bridges that
    # simple thresholding fractures into disconnected fragments.
    t_high = thresh * 0.45
    t_low = thresh * 0.18
    vessel_hyst = apply_hysteresis_threshold(vesselness_norm, t_low, t_high)

    # Apply retina FOV mask (small margin to remove aperture ring)
    retina_mask = _get_retina_mask(img, erode_px=_scale(5, res))
    vessel_hyst[~retina_mask] = False

    # Remove small disconnected noise components
    min_vessel_area = _scale(20, res)
    vessel_uint = vessel_hyst.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_uint, connectivity=8)
    clean_vessel_mask = np.zeros_like(vessel_uint)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_vessel_area:
            clean_vessel_mask[labels == i] = 1

    # Light morphological closing (3x3) to bridge arteriolar light reflex gaps
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_vessel_mask = cv2.morphologyEx(clean_vessel_mask, cv2.MORPH_CLOSE, kernel_bridge)
    clean_vessel_mask[~retina_mask] = 0

    # --- Generate skeleton (1-pixel wide) ---
    vessel_skeleton = skeletonize(clean_vessel_mask > 0)

    # --- Vessel width validation (remove massive non-vessel blobs) ---
    if np.any(clean_vessel_mask > 0):
        dist_transform = cv2.distanceTransform(clean_vessel_mask, cv2.DIST_L2, 5)
        max_vessel_width = _scale(12, res)

        labeled_vessels = label(clean_vessel_mask)
        regions = regionprops(labeled_vessels)
        for props in regions:
            component_skeleton = vessel_skeleton & (labeled_vessels == props.label)
            if np.any(component_skeleton):
                widths = dist_transform[component_skeleton]
                mean_width = np.mean(widths)
                if mean_width > max_vessel_width:
                    clean_vessel_mask[labeled_vessels == props.label] = 0

    # Regenerate skeleton after width filtering
    vessel_skeleton = skeletonize(clean_vessel_mask > 0)

    visual_mask = clean_vessel_mask > 0

    # --- SUPPRESSION MASK: dilated vessel region for aggressive lesion rejection ---
    # A 7x7 dilation around the vessel mask ensures that dark spots sitting on
    # vessel edges, bifurcation shadows, and perivascular contrast artifacts
    # are all suppressed during MA/lesion detection.
    supp_dilate_k = _scale(3, res) * 2 + 1  # ~7px at 512
    suppression_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (supp_dilate_k, supp_dilate_k))
    suppression_mask = cv2.dilate(clean_vessel_mask, suppression_kernel)
    supp_mask = suppression_mask > 0

    return visual_mask, supp_mask, vessel_skeleton


# ============================================================================
# MODULE 3: LESION DETECTION (Microaneurysms + Hard Exudates)
# ============================================================================

def detect_lesions(img, visual_vessel_mask, suppression_vessel_mask, vessel_skeleton,
                   od_mask, retina_mask_eroded, res=DEFAULT_RES):
    """
    Detects Microaneurysms (MAs) and Hard Exudates (EXs).
    """
    ma_mask = _detect_microaneurysms(img, suppression_vessel_mask, vessel_skeleton,
                                      od_mask, retina_mask_eroded, res)
    ex_mask = _detect_exudates(img, suppression_vessel_mask, od_mask, res)

    return ma_mask, ex_mask


def _detect_microaneurysms(img, suppression_vessel_mask, vessel_skeleton,
                            od_mask, retina_mask_eroded, res=DEFAULT_RES):
    """
    MA detection with dilated vessel suppression, skeleton-based proximity
    check, strict circularity, and local contrast validation.

    Key design choices:
      - Suppression mask is already dilated 7px around vessels (from extract_vessels)
        so any dark pixel sitting on or near a vessel edge is rejected outright.
      - Skeleton proximity buffer raised to 4px to reject bifurcation shadows.
      - Eccentricity capped at 0.65 (true MAs are near-perfect circles).
      - Local contrast check requires candidate to be >=12% darker than annular bg.
    """
    g_chan = img[:, :, 1]

    # Bottom-Hat (Black-Hat) filter extracts small dark circular structures
    se_size = _scale(4, res) * 2 + 1  # ~9px disk at 512
    kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    ma_candidates = cv2.morphologyEx(g_chan, cv2.MORPH_BLACKHAT, kernel_ma)

    # Threshold on Black-Hat response
    ma_mask = ma_candidates > 20

    # Suppress using the dilated vessel mask (already 7px wide around vessels)
    ma_mask[suppression_vessel_mask > 0] = False
    ma_mask[od_mask] = False
    ma_mask[~retina_mask_eroded] = False

    labeled_ma = label(ma_mask.astype(np.uint8))
    regions = regionprops(labeled_ma)
    clean_ma_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Distance from vessel skeleton for proximity check
    if np.any(vessel_skeleton):
        skeleton_dist = ndimage.distance_transform_edt(~vessel_skeleton)
    else:
        skeleton_dist = np.ones(img.shape[:2]) * 999

    ma_area_min = max(2, _scale(2, res))
    ma_area_max = _scale(20, res)  # Tighter upper bound -- large blobs are not MAs

    for props in regions:
        area = props.area

        if area < ma_area_min or area > ma_area_max:
            continue

        # STRICT circularity: eccentricity <= 0.65 (true MAs are very round)
        if props.eccentricity > 0.65:
            continue

        coords = props.coords
        centroid_r, centroid_c = int(props.centroid[0]), int(props.centroid[1])

        # Skeleton proximity: reject if centroid is within 4px of vessel centerline
        min_skeleton_dist = skeleton_dist[centroid_r, centroid_c]
        if min_skeleton_dist < 4:
            continue

        # Intensity validation: candidate must be >=12% darker than local background
        rr, cc = coords[:, 0], coords[:, 1]

        ring_outer = _scale(6, res)
        r_min = max(0, int(np.min(rr)) - ring_outer)
        r_max = min(img.shape[0], int(np.max(rr)) + ring_outer + 1)
        c_min = max(0, int(np.min(cc)) - ring_outer)
        c_max = min(img.shape[1], int(np.max(cc)) + ring_outer + 1)

        local_g = g_chan[r_min:r_max, c_min:c_max].astype(float)
        local_mask = np.zeros_like(local_g, dtype=bool)
        local_mask[rr - r_min, cc - c_min] = True

        bg_mask = ~local_mask & (local_g > 0)
        if np.sum(bg_mask) > 5:
            candidate_mean = np.mean(local_g[local_mask])
            bg_mean = np.mean(local_g[bg_mask])
            # 0.88 means candidate must be at least 12% darker than background
            if candidate_mean > bg_mean * 0.88:
                continue

        clean_ma_mask[rr, cc] = 1

    return clean_ma_mask > 0


def _detect_exudates(img, suppression_vessel_mask, od_mask, res=DEFAULT_RES):
    """
    Hard exudate detection using LAB color space with dynamic Otsu.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_chan, a_chan, b_chan = cv2.split(lab)

    L_chan_masked = L_chan.copy()
    b_chan_masked = b_chan.copy()
    L_chan_masked[od_mask] = 0
    b_chan_masked[od_mask] = 128

    thresh_L = cv2.threshold(L_chan_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_b = cv2.threshold(b_chan_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    ex_mask = (L_chan_masked > thresh_L) & (b_chan_masked > thresh_b)
    ex_mask_uint = ex_mask.astype(np.uint8)

    ex_area_min = _scale(3, res)
    ex_area_max = _scale(5000, res)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ex_mask_uint, connectivity=8)
    clean_ex_mask = np.zeros_like(ex_mask_uint)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if ex_area_min < area < ex_area_max:
            clean_ex_mask[labels == i] = 1

    ex_mask = clean_ex_mask > 0
    ex_mask[suppression_vessel_mask] = False

    return ex_mask


# ============================================================================
# MODULE 4: HEMORRHAGE CLASSIFICATION
# ============================================================================

def classify_hemorrhages(img, visual_vessel_mask, suppression_vessel_mask,
                          od_mask, fovea_mask, ma_mask, res=DEFAULT_RES):
    """
    Detects and classifies hemorrhages into Dot-Blot, Flame, and Pre-retinal.
    Uses adaptive thresholding + green/red hemoglobin attenuation validation.
    """
    imgH, imgW = img.shape[:2]
    g_chan = img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)

    retina_mask = _get_retina_mask(img, erode_px=_scale(35, res))

    # --- ADAPTIVE THRESHOLD (replaces fixed g_inv > 180) ---
    retina_pixels = g_inv[retina_mask]
    if len(retina_pixels) == 0:
        empty = np.zeros((imgH, imgW), dtype=bool)
        return empty, empty, empty, empty

    mean_val = np.mean(retina_pixels)
    std_val = np.std(retina_pixels)
    # Relax standard deviation multiplier to catch diffuse large hemorrhages
    # and lower the hard cap because CLAHE doesn't always push hemorrhages to 240+
    adaptive_thresh = min(mean_val + 1.2 * std_val, 190)

    all_dark_anomalies = g_inv > adaptive_thresh

    all_dark_anomalies[suppression_vessel_mask > 0] = False
    all_dark_anomalies[od_mask] = False
    all_dark_anomalies[ma_mask] = False
    all_dark_anomalies[~retina_mask] = False
    all_dark_anomalies[fovea_mask] = False

    # --- GREEN/RED HEMOGLOBIN ATTENUATION VALIDATION ---
    all_dark_anomalies = _validate_hemorrhage_color(img, all_dark_anomalies, retina_mask, res)

    min_hem_area = _scale(25, res)
    all_dark_uint = all_dark_anomalies.astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    all_dark_uint = cv2.morphologyEx(all_dark_uint, cv2.MORPH_OPEN, kernel_open)

    dot_blot_mask = np.zeros((imgH, imgW), dtype=np.uint8)
    flame_mask = np.zeros((imgH, imgW), dtype=np.uint8)
    preretinal_mask = np.zeros((imgH, imgW), dtype=np.uint8)

    labeled_img = label(all_dark_uint)
    regions = regionprops(labeled_img)

    preretinal_min = _scale(1000, res)
    preretinal_max = _scale(20000, res)
    flame_max = _scale(1000, res)

    for props in regions:
        area = props.area
        ecc = props.eccentricity
        coords = props.coords
        rows, cols = coords[:, 0], coords[:, 1]

        if area < min_hem_area:
            continue

        if preretinal_min < area < preretinal_max:
            preretinal_mask[rows, cols] = 1
        elif ecc > 0.85 and min_hem_area < area <= flame_max:
            flame_mask[rows, cols] = 1
        elif ecc <= 0.85 and min_hem_area < area <= flame_max:
            dot_blot_mask[rows, cols] = 1

    dot_blot = dot_blot_mask > 0
    flame = flame_mask > 0
    preretinal = preretinal_mask > 0
    hemorrhage = dot_blot | flame | preretinal

    return dot_blot, flame, preretinal, hemorrhage


def _validate_hemorrhage_color(img, candidate_mask, retina_mask, res=DEFAULT_RES):
    """
    Validate hemorrhage candidates using green/red hemoglobin attenuation.
    
    Updated for Grade 4 Proliferative DR:
    Large hemorrhages significantly darken the local background and also 
    lower the red channel. We use a larger blur kernel for a more global
    background, and relax thresholds for large candidate areas.
    """
    g_chan = img[:, :, 1].astype(float)
    r_chan = img[:, :, 2].astype(float)

    # Use a much larger blur kernel so massive hemorrhages don't completely skew local bg
    blur_k = _scale(75, res)
    blur_k = blur_k * 2 + 1
    g_bg = cv2.GaussianBlur(g_chan, (blur_k, blur_k), 0)
    r_bg = cv2.GaussianBlur(r_chan, (blur_k, blur_k), 0)

    g_bg = np.maximum(g_bg, 1.0)
    r_bg = np.maximum(r_bg, 1.0)

    validated_mask = candidate_mask.copy()

    labeled = label(candidate_mask.astype(np.uint8))
    regions = regionprops(labeled)

    for props in regions:
        coords = props.coords
        rows, cols = coords[:, 0], coords[:, 1]
        
        area = len(rows)

        g_cand = np.mean(g_chan[rows, cols])
        r_cand = np.mean(r_chan[rows, cols])

        g_local_bg = np.mean(g_bg[rows, cols])
        r_local_bg = np.mean(r_bg[rows, cols])

        g_ratio = g_cand / g_local_bg
        r_ratio = r_cand / r_local_bg

        # Dynamic thresholds based on size
        is_large = area > _scale(150, res)
        g_thresh = 0.95 if is_large else 0.88
        r_thresh = 0.65 if is_large else 0.85

        if not (g_ratio < g_thresh and r_ratio > r_thresh):
            validated_mask[rows, cols] = False

    return validated_mask


# ============================================================================
# MODULE 5: FEATURE EXTRACTION (Expanded Feature Vector)
# ============================================================================

def extract_features(img, od_mask, od_center, od_radius, fovea_loc, fovea_mask,
                     visual_vessel_mask, vessel_skeleton,
                     ma_mask, ex_mask,
                     dot_blot_mask, flame_mask, preretinal_mask,
                     res=DEFAULT_RES):
    """
    Compute the expanded feature vector for downstream classification.
    """
    imgH, imgW = img.shape[:2]
    retina_mask = _get_retina_mask(img, erode_px=0)
    retina_area = max(1, np.sum(retina_mask))

    features = {}

    # MA features
    ma_labeled = label(ma_mask.astype(np.uint8))
    features['MA_Count'] = ma_labeled.max()
    features['MA_Density_Pct'] = round((np.sum(ma_mask) / retina_area) * 100, 6)

    # Exudate features
    ex_labeled = label(ex_mask.astype(np.uint8))
    features['EX_Area_Pct'] = round((np.sum(ex_mask) / retina_area) * 100, 4)
    features['EX_Cluster_Count'] = ex_labeled.max()

    if np.any(ex_mask):
        ex_coords = np.argwhere(ex_mask)
        fovea_y, fovea_x = fovea_loc[1], fovea_loc[0]
        distances = np.sqrt((ex_coords[:, 0] - fovea_y)**2 + (ex_coords[:, 1] - fovea_x)**2)
        features['EX_Closest_Fovea_Dist'] = round(float(np.min(distances)), 2)
    else:
        features['EX_Closest_Fovea_Dist'] = float(res)

    # Hemorrhage features
    hemorrhage_mask = dot_blot_mask | flame_mask | preretinal_mask
    features['HEM_DotBlot_Count'] = label(dot_blot_mask.astype(np.uint8)).max()
    features['HEM_Flame_Count'] = label(flame_mask.astype(np.uint8)).max()
    features['HEM_Total_Pct'] = round((np.sum(hemorrhage_mask) / retina_area) * 100, 4)
    features['HEM_PreRetinal_Pct'] = round((np.sum(preretinal_mask) / retina_area) * 100, 4)

    # ETDRS 4-Quadrant Hemorrhage Counts
    q_counts = _compute_quadrant_counts(hemorrhage_mask, fovea_loc, imgH, imgW)
    features['HEM_Q1_Count'] = q_counts[0]
    features['HEM_Q2_Count'] = q_counts[1]
    features['HEM_Q3_Count'] = q_counts[2]
    features['HEM_Q4_Count'] = q_counts[3]

    # Vessel features
    features['Vessel_Coverage_Pct'] = round((np.sum(visual_vessel_mask) / retina_area) * 100, 2)

    if np.any(vessel_skeleton):
        features['Vessel_Tortuosity'] = round(_compute_tortuosity(vessel_skeleton), 4)
    else:
        features['Vessel_Tortuosity'] = 0.0

    if np.any(visual_vessel_mask):
        features['Vessel_Fractal_Dim'] = round(_box_counting_fractal_dim(visual_vessel_mask), 4)
    else:
        features['Vessel_Fractal_Dim'] = 0.0

    # Anatomical features
    features['OD_Radius'] = od_radius
    fovea_x, fovea_y = fovea_loc
    od_x, od_y = od_center
    features['Fovea_OD_Dist'] = round(float(np.sqrt((fovea_x - od_x)**2 + (fovea_y - od_y)**2)), 2)

    return features


def _compute_quadrant_counts(mask, fovea_loc, imgH, imgW):
    """Count hemorrhage components in each ETDRS quadrant."""
    fx, fy = fovea_loc
    labeled = label(mask.astype(np.uint8))
    regions = regionprops(labeled)

    counts = [0, 0, 0, 0]

    for props in regions:
        cy, cx = props.centroid
        if cy < fy and cx >= fx:
            counts[0] += 1
        elif cy < fy and cx < fx:
            counts[1] += 1
        elif cy >= fy and cx < fx:
            counts[2] += 1
        elif cy >= fy and cx >= fx:
            counts[3] += 1

    return counts


def _compute_tortuosity(skeleton):
    """Compute vessel tortuosity as mean arc-to-chord ratio."""
    labeled = label(skeleton.astype(np.uint8))
    regions = regionprops(labeled)

    tortuosities = []
    for props in regions:
        if props.area < 10:
            continue
        coords = props.coords
        arc_length = len(coords)
        p1 = coords[0]
        p2 = coords[-1]
        chord_length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if chord_length > 0:
            tortuosities.append(arc_length / chord_length)

    return np.mean(tortuosities) if tortuosities else 1.0


def _box_counting_fractal_dim(mask):
    """Compute fractal dimension using box-counting method."""
    pixels = np.argwhere(mask)
    if len(pixels) < 10:
        return 0.0

    Lx = np.max(pixels[:, 0]) - np.min(pixels[:, 0]) + 1
    Ly = np.max(pixels[:, 1]) - np.min(pixels[:, 1]) + 1
    L = max(Lx, Ly)

    if L < 4:
        return 0.0

    sizes = []
    counts = []
    s = 2
    while s < L // 2:
        sizes.append(s)
        shifted = pixels - pixels.min(axis=0)
        boxes = set()
        for p in shifted:
            boxes.add((p[0] // s, p[1] // s))
        counts.append(len(boxes))
        s *= 2

    if len(sizes) < 2:
        return 0.0

    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)

    return -coeffs[0]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_retina_mask(img, erode_px=0):
    """Get binary retina mask (non-black region), optionally eroded."""
    r_chan = img[:, :, 2]
    retina_mask = (r_chan > 20).astype(np.uint8)

    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erode_px * 2 + 1, erode_px * 2 + 1))
        retina_mask = cv2.erode(retina_mask, kernel)

    return retina_mask > 0


def run_full_pipeline(img, res=DEFAULT_RES):
    """
    Run the complete Stage 2 pipeline on a single image.

    Args:
        img: BGR image (any size — will be resized to res x res)
        res: Target resolution (default 512)

    Returns:
        dict with all masks, landmarks, and features
    """
    if img.shape[:2] != (res, res):
        img = cv2.resize(img, (res, res))

    # Module 1: Landmarks
    od_mask, od_center, od_radius, fovea_loc, fovea_mask = extract_landmarks(img, res)

    # Module 2: Vessels
    visual_vessel_mask, suppression_vessel_mask, vessel_skeleton = extract_vessels(img, od_mask, res)

    # Retina mask for lesion detection
    retina_mask_eroded = _get_retina_mask(img, erode_px=_scale(15, res))

    # Module 3: Lesions
    ma_mask, ex_mask = detect_lesions(
        img, visual_vessel_mask, suppression_vessel_mask, vessel_skeleton,
        od_mask, retina_mask_eroded, res
    )

    # Module 4: Hemorrhages
    dot_blot_mask, flame_mask, preretinal_mask, hemorrhage_mask = classify_hemorrhages(
        img, visual_vessel_mask, suppression_vessel_mask,
        od_mask, fovea_mask, ma_mask, res
    )

    # Module 5: Feature extraction
    features = extract_features(
        img, od_mask, od_center, od_radius, fovea_loc, fovea_mask,
        visual_vessel_mask, vessel_skeleton,
        ma_mask, ex_mask,
        dot_blot_mask, flame_mask, preretinal_mask, res
    )

    return {
        'img': img,
        'od_mask': od_mask,
        'od_center': od_center,
        'od_radius': od_radius,
        'fovea_loc': fovea_loc,
        'fovea_mask': fovea_mask,
        'visual_vessel_mask': visual_vessel_mask,
        'suppression_vessel_mask': suppression_vessel_mask,
        'vessel_skeleton': vessel_skeleton,
        'ma_mask': ma_mask,
        'ex_mask': ex_mask,
        'dot_blot_mask': dot_blot_mask,
        'flame_mask': flame_mask,
        'preretinal_mask': preretinal_mask,
        'hemorrhage_mask': hemorrhage_mask,
        'features': features,
    }
