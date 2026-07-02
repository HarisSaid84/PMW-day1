import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- 1. Synthetic "heritage arch" structure ----------
def build_arch_points(n_base=14, n_arch=24, width=2.0, height=1.6, radius=1.0, depth=1.2):
    pts = []
    # two vertical side walls (base pillars)
    for z in np.linspace(0, depth, 4):
        for y in np.linspace(0, height, n_base // 2):
            pts.append([-width/2, y, z])
            pts.append([ width/2, y, z])
    # semicircular arch on the front face (z=0) and back face (z=depth)
    for z in [0.0, depth]:
        for theta in np.linspace(0, np.pi, n_arch):
            x = radius * np.cos(theta)
            y = height + radius * np.sin(theta)
            pts.append([x, y, z])
    # scattered "texture" points across the front facade for feature matching
    for _ in range(120):
        x = rng.uniform(-width/2, width/2)
        y = rng.uniform(0, height + radius)
        z = rng.choice([0.0, depth])
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float64)

pts3d = build_arch_points()
print("3D structure points (before spacing filter):", pts3d.shape)

# ---------- 2. Camera intrinsics + two poses (small baseline, slight rotation) ----------
f = 800.0
K = np.array([[f, 0, 320],
              [0, f, 240],
              [0, 0,   1]], dtype=np.float64)

def look_at(cam_pos, target=np.array([0,0.8,0.6]), up=np.array([0,1,0])):
    z = (target - cam_pos); z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ cam_pos
    return R, t

cam1_pos = np.array([-0.35, 0.9, -3.0])
cam2_pos = np.array([ 0.35, 0.95, -2.9])
R1, t1 = look_at(cam1_pos)
R2, t2 = look_at(cam2_pos)

def project(pts, R, t, K):
    cam = (R @ pts.T).T + t
    proj = (K @ cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv, cam[:, 2]

# Greedily drop points that would project too close together in view 1 (avoids
# ambiguous texture-patch overlap and unreliable ground-truth matching later on).
uv1_pre, _ = project(pts3d, R1, t1, K)
min_spacing = 16.0
keep = []
kept_uv = []
order_idx = np.arange(len(pts3d))
rng.shuffle(order_idx)
for i in order_idx:
    uv = uv1_pre[i]
    if not (0 <= uv[0] < 640 and 0 <= uv[1] < 480):
        continue
    if all(np.linalg.norm(uv - k) > min_spacing for k in kept_uv):
        keep.append(i)
        kept_uv.append(uv)
pts3d = pts3d[np.array(keep)]
print("3D structure points (after spacing filter):", pts3d.shape)

uv1, depth1 = project(pts3d, R1, t1, K)
uv2, depth2 = project(pts3d, R2, t2, K)
print("Projected point ranges (view 1):", uv1.min(0), uv1.max(0))

# ---------- 3. Render two synthetic images (consistent per-point texture patches) ----------
patch_size = 9
half = patch_size // 2
# one fixed random texture patch per 3D point -> identical surface appearance in both views
patches = [rng.integers(50, 220, (patch_size, patch_size), dtype=np.uint8) for _ in range(len(pts3d))]

def render(uv, depths, size=(480, 640)):
    img = np.full((*size, 3), 18, dtype=np.uint8)
    order = np.argsort(-depths)  # paint far points first so near points occlude
    for idx in order:
        x, y = int(round(uv[idx][0])), int(round(uv[idx][1]))
        x0, x1 = x - half, x + half + 1
        y0, y1 = y - half, y + half + 1
        if x0 < 0 or y0 < 0 or x1 > size[1] or y1 > size[0]:
            continue
        shade = np.clip(255 - depths[idx]*22, 60, 255)
        patch = (patches[idx].astype(np.float32) * (shade/255.0)).astype(np.uint8)
        for c, mult in zip(range(3), (1.0, 0.82, 0.55)):  # warm sandstone tint
            img[y0:y1, x0:x1, c] = np.clip(patch.astype(np.float32) * mult, 0, 255).astype(np.uint8)
    return img

img1 = render(uv1, depth1)
img2 = render(uv2, depth2)
cv2.imwrite("/home/claude/pmw_repro/view1.png", img1)
cv2.imwrite("/home/claude/pmw_repro/view2.png", img2)
print("Rendered two synthetic views.")

# ---------- 4. ORB feature detection + matching ----------
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
print(f"Keypoints -> view1: {len(kp1)}, view2: {len(kp2)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda m: m.distance)
print(f"Raw ORB matches: {len(matches)}")

match_vis = cv2.drawMatches(img1, kp1, img2, kp2, matches[:60], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("/home/claude/pmw_repro/matches.png", match_vis)

# ---------- 5. Essential matrix + pose recovery from ORB-matched points ----------
pts1_matched = np.float64([kp1[m.queryIdx].pt for m in matches])
pts2_matched = np.float64([kp2[m.trainIdx].pt for m in matches])

E, mask = cv2.findEssentialMat(pts1_matched, pts2_matched, K, method=cv2.RANSAC,
                                prob=0.999, threshold=1.0)
inliers = int(mask.sum()) if mask is not None else 0
print(f"Essential matrix inliers: {inliers} / {len(matches)}")

_, R_est, t_est, mask_pose = cv2.recoverPose(E, pts1_matched, pts2_matched, K, mask=mask)

# ---------- 6. Triangulate matched points & compare against ground truth ----------
P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
P2 = K @ np.hstack([R_est, t_est])
inlier_mask = mask_pose.ravel().astype(bool)
p1_in = pts1_matched[inlier_mask].T
p2_in = pts2_matched[inlier_mask].T
pts4d = cv2.triangulatePoints(P1, P2, p1_in, p2_in)
pts3d_est = (pts4d[:3] / pts4d[3]).T
print(f"Triangulated {pts3d_est.shape[0]} inlier points from ORB matches.")

# Map each matched keypoint back to the *nearest* ground-truth projected point,
# so we can quantify reconstruction error against the known synthetic structure.
uv1_inlier = pts1_matched[inlier_mask]
nn_idx = []
for uv in uv1_inlier:
    d = np.linalg.norm(uv1 - uv, axis=1)
    nn_idx.append(np.argmin(d))
nn_idx = np.array(nn_idx)
gt_matched = pts3d[nn_idx]
# ground truth in camera-1 frame (since our triangulated cloud is expressed there)
gt_cam1 = (R1 @ gt_matched.T).T + t1

# Recover unknown global scale (monocular reconstruction is scale-ambiguous)
num = np.sum(np.linalg.norm(gt_cam1, axis=1) * np.linalg.norm(pts3d_est, axis=1))
den = np.sum(np.linalg.norm(pts3d_est, axis=1) ** 2)
scale = num / den
pts3d_est_scaled = pts3d_est * scale

err = np.linalg.norm(pts3d_est_scaled - gt_cam1, axis=1)
print(f"Recovered scale factor: {scale:.4f}")
print(f"Mean reconstruction error: {err.mean():.4f} (scene units, arch ~2m wide)")
print(f"Median reconstruction error: {np.median(err):.4f}")

# ---------- 7. Plot ground truth vs reconstructed point cloud ----------
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(gt_cam1[:,0], gt_cam1[:,1], gt_cam1[:,2], c="#5FB3A3", s=14, label="ground truth")
ax.scatter(pts3d_est_scaled[:,0], pts3d_est_scaled[:,1], pts3d_est_scaled[:,2],
           c="#E8A33D", s=14, label="triangulated (ORB + SfM)")
ax.set_title("Synthetic arch: ground truth vs. reconstructed points")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/pmw_repro/reconstruction_compare.png", dpi=140)
print("Saved reconstruction comparison plot.")

# ---------- 8. Stereo depth map demo (rectified stereo pair, StereoSGBM) ----------
def render_stereo_scene(shift=0):
    size = (480, 640)
    img = np.full((*size, 3), 30, dtype=np.uint8)
    # background wall (far)
    cv2.rectangle(img, (0,0), (640,480), (40,45,42), -1)
    # mid-depth "facade" rectangle
    cv2.rectangle(img, (120-shift, 100), (520-shift, 420), (70,90,85), -1)
    cv2.rectangle(img, (120-shift, 100), (520-shift, 420), (20,25,22), 3)
    # near "pillar" object, shifts more with camera motion (closer = larger disparity)
    cv2.rectangle(img, (260-2*shift, 220), (380-2*shift, 420), (150,170,120), -1)
    # add mild texture/noise so SGBM has something to match
    noise = (rng.normal(0, 6, img.shape)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

left = render_stereo_scene(shift=0)
right = render_stereo_scene(shift=14)  # 14px disparity baseline for the far facade
cv2.imwrite("/home/claude/pmw_repro/stereo_left.png", left)
cv2.imwrite("/home/claude/pmw_repro/stereo_right.png", right)

grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7,
                                P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1,
                                uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
disparity[disparity <= 0] = 0.01
baseline_m, focal_px = 0.06, 800.0
depth_map = (focal_px * baseline_m) / disparity

fig, axes = plt.subplots(1, 3, figsize=(13,4))
axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)); axes[0].set_title("left"); axes[0].axis("off")
axes[1].imshow(disparity, cmap="viridis"); axes[1].set_title("disparity (SGBM)"); axes[1].axis("off")
im = axes[2].imshow(depth_map, cmap="magma", vmax=np.percentile(depth_map,95))
axes[2].set_title("depth map (m)"); axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig("/home/claude/pmw_repro/stereo_depth.png", dpi=140)
print("Saved stereo depth map figure.")
print("DONE")
