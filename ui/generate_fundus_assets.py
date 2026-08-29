import os
import cv2
import numpy as np

def generate_fundus_assets(output_dir="ui/public/assets"):
    os.makedirs(output_dir, exist_ok=True)
    size = 400
    center = (size // 2, size // 2)
    radius = int(size * 0.46)
    
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist <= radius

    def create_retina():
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if mask[i, j]:
                    d = dist[i, j] / radius
                    img[i, j] = [
                        int(18 + 10 * (1 - d)),
                        int(65 + 30 * (1 - d)),
                        int(190 + 35 * (1 - d))
                    ]
        # Optic Disc (Right / Nasal)
        od_center = (int(size * 0.72), int(size * 0.50))
        cv2.ellipse(img, od_center, (30, 38), 0, 0, 360, (120, 220, 255), -1)
        cv2.ellipse(img, od_center, (18, 24), 0, 0, 360, (160, 240, 255), -1)
        
        # Macula (Left / Temporal)
        macula_center = (int(size * 0.35), int(size * 0.52))
        cv2.circle(img, macula_center, 26, (12, 48, 145), -1)
        cv2.circle(img, macula_center, 6, (6, 32, 115), -1)
        
        # Vessels
        pts_sup = np.array([[od_center[0], od_center[1]], [int(size*0.55), int(size*0.25)], [int(size*0.25), int(size*0.30)], [int(size*0.12), int(size*0.40)]], np.int32)
        pts_inf = np.array([[od_center[0], od_center[1]], [int(size*0.55), int(size*0.75)], [int(size*0.25), int(size*0.70)], [int(size*0.12), int(size*0.60)]], np.int32)
        cv2.polylines(img, [pts_sup], False, (15, 30, 95), 4, cv2.LINE_AA)
        cv2.polylines(img, [pts_inf], False, (15, 30, 95), 4, cv2.LINE_AA)
        
        for (s_pt, e_pt) in [
            ((int(size*0.55), int(size*0.25)), (int(size*0.45), int(size*0.16))),
            ((int(size*0.40), int(size*0.28)), (int(size*0.38), int(size*0.42))),
            ((int(size*0.55), int(size*0.75)), (int(size*0.45), int(size*0.84))),
            ((int(size*0.40), int(size*0.72)), (int(size*0.38), int(size*0.58))),
        ]:
            cv2.line(img, s_pt, e_pt, (18, 35, 105), 2, cv2.LINE_AA)
            
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img[~mask] = 0
        return img

    # --- 1. Grade 2 (Moderate NPDR - Real Patient Preset) ---
    g2_raw = create_retina()
    ma_coords_g2 = [
        (165, 140), (180, 155), (135, 160), (145, 235),
        (225, 255), (115, 185), (210, 180), (195, 245),
        (160, 255), (245, 175), (215, 230), (130, 220)
    ]
    for pt in ma_coords_g2:
        cv2.circle(g2_raw, pt, 3, (8, 12, 75), -1)
    he_coords_g2 = [(185, 125), (192, 128), (198, 122), (188, 132)]
    for pt in he_coords_g2:
        cv2.circle(g2_raw, pt, 2, (170, 240, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "grade2_raw.png"), g2_raw)

    # G2 Preprocessed
    lab = cv2.cvtColor(g2_raw, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    g2_enh = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    g2_enh[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "grade2_preprocessed.png"), g2_enh)

    # G2 Lesions
    g2_les = g2_enh.copy()
    for pt in ma_coords_g2:
        cv2.circle(g2_les, pt, 7, (0, 0, 230), 2, cv2.LINE_AA)
    for pt in he_coords_g2:
        cv2.circle(g2_les, pt, 5, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "grade2_lesions.png"), g2_les)

    # G2 Grad-CAM (Heatmap Only & Blended)
    cam_g2_raw = np.zeros((size, size), dtype=np.float32)
    for pt in ma_coords_g2 + he_coords_g2:
        cv2.circle(cam_g2_raw, pt, 32, 1.0, -1)
    cam_g2_raw = cv2.GaussianBlur(cam_g2_raw, (45, 45), 0)
    cam_g2_raw = (cam_g2_raw / (cam_g2_raw.max() + 1e-6) * 255).astype(np.uint8)
    cam_g2_color = cv2.applyColorMap(cam_g2_raw, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "grade2_heatmap.png"), cam_g2_color)
    g2_cam_blended = cv2.addWeighted(g2_enh, 0.60, cam_g2_color, 0.40, 0)
    g2_cam_blended[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "grade2_gradcam.png"), g2_cam_blended)

    # --- 2. Grade 0 (Normal Eye Preset) ---
    g0_raw = create_retina()
    cv2.imwrite(os.path.join(output_dir, "grade0_raw.png"), g0_raw)
    cv2.imwrite(os.path.join(output_dir, "grade0_preprocessed.png"), g0_raw)
    cv2.imwrite(os.path.join(output_dir, "grade0_lesions.png"), g0_raw)
    g0_cam = cv2.applyColorMap(np.zeros((size, size), dtype=np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "grade0_heatmap.png"), g0_cam)
    cv2.imwrite(os.path.join(output_dir, "grade0_gradcam.png"), g0_raw)

    # --- 3. Grade 3 (Severe NPDR Preset) ---
    g3_raw = create_retina()
    # Extensive hemorrhages across all quadrants
    hem_coords = [
        (120, 110), (140, 130), (160, 100), (110, 170), (130, 200), (150, 250),
        (220, 120), (250, 140), (210, 260), (240, 270), (270, 230), (180, 280),
        (90, 220), (170, 310), (230, 300), (280, 180), (100, 150), (260, 100)
    ]
    for pt in hem_coords:
        cv2.circle(g3_raw, pt, 6, (5, 5, 50), -1)
        cv2.circle(g3_raw, (pt[0]+2, pt[1]+2), 4, (5, 10, 70), -1)
    cv2.imwrite(os.path.join(output_dir, "grade3_raw.png"), g3_raw)

    g3_enh = cv2.cvtColor(g3_raw, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(g3_enh)
    g3_enh = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    g3_enh[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "grade3_preprocessed.png"), g3_enh)

    g3_les = g3_enh.copy()
    for pt in hem_coords:
        cv2.rectangle(g3_les, (pt[0]-7, pt[1]-7), (pt[0]+7, pt[1]+7), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "grade3_lesions.png"), g3_les)

    cam_g3_raw = np.zeros((size, size), dtype=np.float32)
    for pt in hem_coords:
        cv2.circle(cam_g3_raw, pt, 35, 1.0, -1)
    cam_g3_raw = cv2.GaussianBlur(cam_g3_raw, (51, 51), 0)
    cam_g3_raw = (cam_g3_raw / (cam_g3_raw.max() + 1e-6) * 255).astype(np.uint8)
    cam_g3_color = cv2.applyColorMap(cam_g3_raw, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, "grade3_heatmap.png"), cam_g3_color)
    g3_cam_blended = cv2.addWeighted(g3_enh, 0.55, cam_g3_color, 0.45, 0)
    g3_cam_blended[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "grade3_gradcam.png"), g3_cam_blended)

    # --- 4. Blurry Ungradable Scan (IQA Rejection Demo) ---
    un_blur = cv2.GaussianBlur(g2_raw, (39, 39), 0)
    dark_factor = np.linspace(0.15, 0.75, size).reshape(1, size)
    un_blur = (un_blur.astype(np.float32) * dark_factor[:, :, None]).astype(np.uint8)
    un_blur[~mask] = 0
    cv2.imwrite(os.path.join(output_dir, "ungradable_raw.png"), un_blur)

    print("All preset assets created successfully in:", output_dir)

if __name__ == "__main__":
    generate_fundus_assets()
