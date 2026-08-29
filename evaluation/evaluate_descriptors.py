"""
Evaluate loop closure descriptors (ResNet18, EfficientNet-B0, DINO ViT-S/8)
on TUM fr1/desk using ground truth poses to determine true loop closure pairs.

For each pair of frames (i, j) with i < j - MIN_GAP:
  - True loop closure if ground truth positions are within POS_THRESHOLD metres
  - Predicted loop closure if descriptor cosine similarity > tau

Outputs precision/recall/F1 per descriptor at tau=0.85, plus a full P-R curve.
"""
import os, sys, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as tv_models

DATASET = os.path.expanduser("~/rgbd_dataset_freiburg1_desk")
RGB_LIST = os.path.join(DATASET, "rgb.txt")
GT_FILE = os.path.join(DATASET, "groundtruth.txt")

MIN_GAP = 30          # frames apart to be eligible as loop closure (avoid trivial matches)
POS_THRESHOLD = 0.30  # metres - within this distance = true loop closure
SAMPLE_STRIDE = 5      # evaluate every 5th frame to keep runtime reasonable
TAU = 0.85

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_rgb_list():
    frames = []
    with open(RGB_LIST) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            frames.append((float(parts[0]), parts[1]))
    return frames

def load_groundtruth():
    gt = []
    with open(GT_FILE) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            t = float(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            gt.append((t, xyz))
    return gt

def nearest_gt(t, gt_list):
    best = min(gt_list, key=lambda g: abs(g[0] - t))
    return best[1]

def build_models():
    models = {}

    r18 = tv_models.resnet18(pretrained=True)
    r18 = torch.nn.Sequential(*list(r18.children())[:-1]).to(device).eval()
    models['ResNet18'] = lambda t: r18(t).squeeze()

    import timm
    eff = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0).to(device).eval()
    models['EfficientNet-B0'] = lambda t: eff(t).squeeze()

    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()
    models['DINO-ViT-S8'] = lambda t: dino(t).squeeze()

    return models

def extract_descriptor(model_fn, frame_grey):
    frame_rgb = cv2.cvtColor(frame_grey, cv2.COLOR_GRAY2RGB)
    tensor = transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        desc = model_fn(tensor)
    return F.normalize(desc, dim=0)

def main():
    print("Loading frame list and ground truth...")
    frames = load_rgb_list()[::SAMPLE_STRIDE]
    gt_list = load_groundtruth()
    print(f"Evaluating on {len(frames)} sampled frames (stride={SAMPLE_STRIDE})")

    print("Loading models (ResNet18, EfficientNet-B0, DINO ViT-S/8)...")
    models = build_models()

    results = {}

    for name, model_fn in models.items():
        print(f"\n=== {name} ===")
        descriptors = []
        latencies = []

        for idx, (t, rgb_path) in enumerate(frames):
            img_path = os.path.join(DATASET, rgb_path)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            t0 = time.time()
            desc = extract_descriptor(model_fn, frame)
            latencies.append((time.time() - t0) * 1000)
            descriptors.append((idx, t, desc))
            if idx % 50 == 0:
                print(f"  processed {idx}/{len(frames)}")

        y_true, y_score = [], []
        N = len(descriptors)
        for i in range(N):
            idx_i, t_i, desc_i = descriptors[i]
            pos_i = nearest_gt(t_i, gt_list)
            for j in range(i + 1, N):
                idx_j, t_j, desc_j = descriptors[j]
                if idx_j - idx_i < MIN_GAP:
                    continue
                pos_j = nearest_gt(t_j, gt_list)
                dist = np.linalg.norm(pos_i - pos_j)
                true_loop = 1 if dist < POS_THRESHOLD else 0
                score = torch.dot(desc_i, desc_j).item()
                y_true.append(true_loop)
                y_score.append(score)

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        print(f"  Score stats: min={y_score.min():.3f} max={y_score.max():.3f} mean={y_score.mean():.3f} std={y_score.std():.3f}")
        if y_true.sum() > 0:
            print(f"  Score at TRUE loops: mean={y_score[y_true==1].mean():.3f} min={y_score[y_true==1].min():.3f} max={y_score[y_true==1].max():.3f}")
        print(f"  Score at non-loops: mean={y_score[y_true==0].mean():.3f}")

        # --- Threshold sweep: find each descriptor's own optimal F1 threshold ---
        thresholds = np.linspace(y_score.min(), y_score.max(), 200)
        best_f1, best_tau, best_p, best_r = 0, TAU, 0, 0
        for th in thresholds:
            y_pred_th = (y_score > th).astype(int)
            tp_ = np.sum((y_pred_th == 1) & (y_true == 1))
            fp_ = np.sum((y_pred_th == 1) & (y_true == 0))
            fn_ = np.sum((y_pred_th == 0) & (y_true == 1))
            p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0
            if f1_ > best_f1:
                best_f1, best_tau, best_p, best_r = f1_, th, p_, r_

        # Simple AUC via trapezoidal rule over sorted thresholds (ROC-style, desc order)
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        tpr_list, fpr_list = [0.0], [0.0]
        tp_c, fp_c = 0, 0
        for label in y_sorted:
            if label == 1:
                tp_c += 1
            else:
                fp_c += 1
            tpr_list.append(tp_c / n_pos if n_pos > 0 else 0)
            fpr_list.append(fp_c / n_neg if n_neg > 0 else 0)
        auc = np.trapz(tpr_list, fpr_list)

        print(f"  Fixed tau=0.85 -> reported above")
        print(f"  BEST threshold={best_tau:.3f} -> Precision={best_p:.3f} Recall={best_r:.3f} F1={best_f1:.3f}")
        print(f"  AUC (threshold-independent) = {auc:.3f}")

        y_pred = (y_score > TAU).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[name] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'n_true_loops': int(np.sum(y_true)),
            'n_pairs': len(y_true),
            'mean_latency_ms': np.mean(latencies),
            'y_true': y_true, 'y_score': y_score,
        }

        print(f"  True loop closure pairs: {int(np.sum(y_true))} / {len(y_true)}")
        print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")

    np.savez(os.path.expanduser('~/mono-slam-cnn-loop-closure/evaluation/results.npz'),
              **{f"{k}_{m}": v for k, res in results.items() for m, v in res.items()
                 if m in ('y_true', 'y_score')})

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Descriptor':<18} {'Precision':<10} {'Recall':<10} {'F1':<8} {'Latency(ms)':<12} {'TruePairs':<10}")
    for name, r in results.items():
        print(f"{name:<18} {r['precision']:<10.3f} {r['recall']:<10.3f} {r['f1']:<8.3f} {r['mean_latency_ms']:<12.2f} {r['n_true_loops']:<10}")

    return results

if __name__ == "__main__":
    main()
