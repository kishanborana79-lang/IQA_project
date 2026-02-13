# regression.py
import os
import zipfile
import shutil
import numpy as np
import cv2
from brisque import BRISQUE

# ---------------------------
# Robust image reader (Windows safe)
# ---------------------------
def read_image_any(path: str):
    """
    Reads image even if cv2.imread fails on Windows paths.
    Returns BGR image or None.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception:
        return None


# ---------------------------
# BRISQUE
# ---------------------------
_brisque = BRISQUE()

def brisque_score_from_img(img_bgr) -> float:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return float(_brisque.score(rgb))

def brisque_score_from_path(img_path: str):
    img = read_image_any(img_path)
    if img is None:
        return None
    return brisque_score_from_img(img)


# ---------------------------
# Classic quality features
# ---------------------------
def _to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def calc_blur_laplacian(img_bgr) -> float:
    gray = _to_gray(img_bgr)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())  # higher = sharper

def calc_brightness(img_bgr) -> float:
    gray = _to_gray(img_bgr)
    return float(gray.mean())  # 0..255

def calc_contrast(img_bgr) -> float:
    gray = _to_gray(img_bgr)
    return float(gray.std())   # higher = more contrast

def calc_noise_est(img_bgr) -> float:
    gray = _to_gray(img_bgr).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    resid = gray - blur
    return float(np.std(resid))

def calc_edge_density(img_bgr) -> float:
    gray = _to_gray(img_bgr)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.mean(edges > 0))  # 0..1


# ---------------------------
# Decision by BRISQUE
# ---------------------------
def decide_by_brisque(b: float) -> str:
    # Tune later if needed
    if b <= 30:
        return "accept"
    elif b <= 45:
        return "borderline"
    return "reject"


# ---------------------------
# Score out of 100 (higher better)
# ---------------------------
def quality_score_100(brisque, blur_var, brightness, contrast, noise):
    # penalties 0 good -> 1 bad
    p_brisque = min(brisque / 60.0, 1.0)                  # lower better
    p_blur = 0.0 if blur_var >= 150 else (1.0 - blur_var / 150.0)

    # brightness ideal ~ 100-160 (center 130)
    if 100 <= brightness <= 160:
        p_bright = 0.0
    else:
        p_bright = min(abs(brightness - 130) / 130.0, 1.0)

    p_contrast = 0.0 if contrast >= 30 else (1.0 - contrast / 30.0)  # low contrast penalize
    p_noise = min(noise / 20.0, 1.0)                                 # higher noise worse

    penalty = (
        0.40 * p_brisque +
        0.20 * p_blur +
        0.15 * p_bright +
        0.15 * p_contrast +
        0.10 * p_noise
    )
    score = 100.0 * (1.0 - penalty)
    return float(max(0.0, min(100.0, score)))


# ---------------------------
# Explain issues
# ---------------------------
def explain_issues(blur_var, brightness, contrast, noise):
    reasons = []
    if blur_var < 80:
        reasons.append("High blur / low sharpness")
    if brightness < 60:
        reasons.append("Too dark")
    if brightness > 200:
        reasons.append("Too bright / overexposed")
    if contrast < 25:
        reasons.append("Low contrast")
    if noise > 12:
        reasons.append("High noise/grain")

    if not reasons:
        reasons.append("Looks fine overall")
    return reasons


# ---------------------------
# Extract all features from image
# ---------------------------
def extract_all_features(img_bgr) -> dict:
    h, w = img_bgr.shape[:2]

    blur_var = calc_blur_laplacian(img_bgr)
    brightness = calc_brightness(img_bgr)
    contrast = calc_contrast(img_bgr)
    noise = calc_noise_est(img_bgr)
    edge_density = calc_edge_density(img_bgr)
    brisque = brisque_score_from_img(img_bgr)

    score100 = quality_score_100(brisque, blur_var, brightness, contrast, noise)

    return {
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "brisque": brisque,
        "blur_var": blur_var,
        "brightness": brightness,
        "contrast": contrast,
        "noise": noise,
        "edge_density": edge_density,
        "score_100": score100,
    }


# ---------------------------
# Evaluate single image (for UI tab-1)
# ---------------------------
def evaluate_image(img_path: str) -> dict:
    img = read_image_any(img_path)
    if img is None:
        return {"path": img_path, "error": "Could not read image"}

    feats = extract_all_features(img)
    decision = decide_by_brisque(feats["brisque"])
    reasons = explain_issues(feats["blur_var"], feats["brightness"], feats["contrast"], feats["noise"])

    return {
        "path": img_path,
        "decision": decision,
        "main_issue": reasons[0],
        "reasons": reasons,
        "score_100": round(feats["score_100"], 2),
        "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats.items()}
    }


# ---------------------------
# Sort ZIP dataset (for UI tab-2)
# ---------------------------
def sort_zip_dataset(zip_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(work_dir)

    acc_dir = os.path.join(out_dir, "accept")
    bor_dir = os.path.join(out_dir, "borderline")
    rej_dir = os.path.join(out_dir, "reject")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(bor_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)

    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    rows = ["filename,brisque,blur_var,brightness,contrast,noise,score_100,decision"]
    counts = {"accept": 0, "borderline": 0, "reject": 0, "unreadable": 0}

    for root, _, files in os.walk(work_dir):
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue

            src = os.path.join(root, f)
            img = read_image_any(src)

            if img is None:
                counts["unreadable"] += 1
                continue

            feats = extract_all_features(img)
            decision = decide_by_brisque(feats["brisque"])
            counts[decision] += 1

            rows.append(
                f"{f},{feats['brisque']:.6f},{feats['blur_var']:.6f},{feats['brightness']:.6f},"
                f"{feats['contrast']:.6f},{feats['noise']:.6f},{feats['score_100']:.2f},{decision}"
            )

            dst_dir = acc_dir if decision == "accept" else bor_dir if decision == "borderline" else rej_dir
            dst = os.path.join(dst_dir, f)

            if os.path.exists(dst):
                base, ext = os.path.splitext(f)
                dst = os.path.join(dst_dir, f"{base}_dup{ext}")

            shutil.copy2(src, dst)

    report_path = os.path.join(out_dir, "report.csv")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(rows))

    shutil.rmtree(work_dir, ignore_errors=True)

    return {"out_dir": out_dir, "report": report_path, "counts": counts}
