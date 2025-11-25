import os, sys, time, glob, cv2, json, numpy as np, torch
from collections import defaultdict, OrderedDict
import torchvision.transforms as T
from torchvision import models
import torch.nn.functional as F
from ultralytics import YOLO
import random
BASE_DIR = "/media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT"
PATHS = {
    "ANNOTATED_DIR": os.path.join(BASE_DIR, "data", "hybrid_data_test"),
    "SAM2_CKPT": os.path.join(BASE_DIR, "output", "sam2_finetuned_final.pth"),
    "SAM2_CONFIG": os.path.join(BASE_DIR, "configs", "sam2.1", "sam2.1_hiera_b+.yaml"),
}
for k, v in PATHS.items():
    print(f"[{'OK' if os.path.exists(v) else 'MISS'}] {k}: {v}")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True
print("[INFO] device:", device)

PROJECT_ROOT = BASE_DIR
SAM2_ROOT = os.path.join(PROJECT_ROOT, "sam2")
for p in [SAM2_ROOT, os.path.join(SAM2_ROOT, "sam2")]:
    if p not in sys.path: sys.path.insert(0, p)

from build_sam import build_sam2
from sam2_image_predictor import SAM2ImagePredictor

def load_sam2_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    info = {}
    if isinstance(ckpt, dict):
        info['classes'] = ckpt.get("classes", [])
        sd = ckpt.get("model_state_dict", ckpt)
    else:
        sd = ckpt
    new_sd = OrderedDict((k[7:], v) if k.startswith("module.") else (k, v) for k, v in sd.items())
    return new_sd, info

def build_backbone(pretrained=True):
    try:
        if hasattr(models, 'resnet18') and pretrained:
            backbone = models.resnet18(weights=getattr(models, "ResNet18_Weights", None).DEFAULT) if hasattr(models, "ResNet18_Weights") else models.resnet18(pretrained=True)
        else:
            backbone = models.resnet18(weights=None)
    except Exception:
        backbone = models.resnet18(weights=None)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1])).to(device).eval()
    return backbone

preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_prototypes(annotated_dir, classes_in_ckpt, backbone, max_per_class=100):
    coco_json = os.path.join(annotated_dir, "hybrid_coco.json")
    if not os.path.exists(coco_json):
        print("[ERR] hybrid_coco.json NOT FOUND")
        return {}

    with open(coco_json, "r") as f:
        data = json.load(f)

    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    class_to_images = defaultdict(list)
    image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    for ann in data["annotations"]:
        cid = ann["category_id"]
        cname = id_to_name[cid]

        if cname not in classes_in_ckpt:
            continue

        img_file = os.path.join(annotated_dir, image_id_to_file[ann["image_id"]])
        if os.path.exists(img_file):
            class_to_images[cname].append(img_file)

    prototypes = {}
    for cls, files in class_to_images.items():
        embs = []
        for p in files[:max_per_class]:
            img = cv2.imread(p)
            if img is None: continue

            inp = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(inp).squeeze().cpu().numpy()
                embs.append(emb)

        if len(embs) > 0:
            prototypes[cls] = np.mean(np.stack(embs, 0), axis=0)

    print(f"[INFO] Built prototypes for {len(prototypes)} / {len(classes_in_ckpt)} classes")
    return prototypes



def generate_grid_boxes(w, h, scales=(0.15, 0.3, 0.45, 0.6), stride=150):
    boxes = []
    for s in scales:
        bw = int(w * s); bh = int(h * s)
        if bw < 32: bw = 32
        if bh < 32: bh = 32
        step_x = max(int(bw * 0.6), stride)
        step_y = max(int(bh * 0.6), stride)
        for x in range(0, w - bw + 1, step_x):
            for y in range(0, h - bh + 1, step_y):
                boxes.append([x, y, x + bw, y + bh])
    return np.array(boxes)

def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0, x2 - x1); ih = max(0, y2 - y1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter + 1e-8
    return inter / union

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def nms_boxes(boxes, scores, iou_thresh=0.3):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        rest = idxs[1:]
        rem = []
        for j in rest:
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                continue
            rem.append(j)
        idxs = np.array(rem, dtype=int)
    return keep

class SAM2Detector:
    def __init__(self, ckpt_path, cfg_path):
        sd, info = load_sam2_weights(ckpt_path)
        self.classes = info.get("classes", [])
        print(f"[INFO] CKPT classes count: {len(self.classes)}")
        self.model = build_sam2(cfg_path, None, device)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.float().to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)

        self.backbone = build_backbone(pretrained=True)
        self.prototypes = build_prototypes(PATHS["ANNOTATED_DIR"], self.classes, self.backbone, max_per_class=100)
        for k in list(self.prototypes.keys()):
            v = self.prototypes[k].ravel()
            self.prototypes[k] = v / (np.linalg.norm(v) + 1e-8)

    def _match_class(self, crop):
        inp = preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.backbone(inp).squeeze().cpu().numpy().ravel()
        qn = q / (np.linalg.norm(q) + 1e-8)
        best_cls = None; best_sc = -1.0
        for cls, proto in self.prototypes.items():
            sc = float(np.dot(qn, proto))
            if sc > best_sc:
                best_sc = sc; best_cls = cls
        return best_cls, best_sc, qn

    @torch.inference_mode()
    def infer_frame(self, frame,
                    box_sc_th=0.35,
                    mask_area_th=600,
                    proto_sim_th=0.6,
                    stride=200,
                    scales=(0.18, 0.3, 0.45)):
        

        H, W = frame.shape[:2]
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale_factor = 0.6
        small_H, small_W = int(H * scale_factor), int(W * scale_factor)
        small_frame = cv2.resize(rgb_small, (small_W, small_H))

        grid_boxes = []
        for s in scales:
            bw, bh = int(small_W * s), int(small_H * s)
            for x in range(0, small_W - bw, stride):
                for y in range(0, small_H - bh, stride):
                    crop = small_frame[y:y + bh, x:x + bw]
                    if crop.size == 0:
                        continue
                    if np.var(crop) < 20:
                        continue
                    grid_boxes.append([x, y, x + bw, y + bh])
        if not grid_boxes:
            return []

        boxes = np.array(grid_boxes, dtype=np.int32)
        if len(boxes) > 60:
            idx = np.random.choice(len(boxes), 60, replace=False)
            boxes = boxes[idx]

        detections = []
        self.predictor.set_image(small_frame)

        batch_size = 10
        for i in range(0, len(boxes), batch_size):
            sub_boxes = boxes[i:i + batch_size]
            try:
                out = self.predictor.predict(box=sub_boxes, multimask_output=False)
                if isinstance(out, tuple):
                    masks, scores = out[:2]
                else:
                    continue
                masks = np.asarray(masks)
                scores = np.asarray(scores)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
            except Exception:
                continue

            for j, sc in enumerate(scores):
                if float(sc) < 0.20:
                    continue
                mask = masks[j].astype(np.uint8)
                area = int(mask.sum())
                if area < 300:
                    continue
                ys, xs = np.where(mask > 0)
                if ys.size == 0:
                    continue
                bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bx1, by1, bx2, by2 = map(lambda z: int(z / scale_factor), [bx1, by1, bx2, by2])
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(W - 1, bx2), min(H - 1, by2)

                crop = frame[by1:by2, bx1:bx2]
                if crop.size == 0:
                    continue
                cls_name, sim, emb = self._match_class(crop)

                if sim < proto_sim_th:
                    continue



                detections.append({
                    "box": [bx1, by1, bx2, by2],
                    "score": float(sc),
                    "cls": cls_name,
                    "sim": float(sim),
                    "emb": emb
                })

        if not detections:
            return []
        boxes_arr = np.array([d["box"] for d in detections], dtype=np.int32)
        scores_arr = np.array([d["score"] * d["sim"] for d in detections], dtype=np.float32)
        keep_idx = nms_boxes(boxes_arr, scores_arr, iou_thresh=0.4)
        return [detections[i] for i in keep_idx]



class SimpleTracker:
    def __init__(self, max_age=15, iou_thresh=0.3, sim_thresh=0.7, lambda_sim=0.5):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.sim_thresh = sim_thresh
        self.lambda_sim = lambda_sim

    def update(self, dets):
        matched = set()
        for tid, track in list(self.tracks.items()):
            track["age"] += 1
            if track["age"] > self.max_age:
                del self.tracks[tid]
                continue
            best_score = 0
            best_det_idx = -1
            for i, d in enumerate(dets):
                if i in matched: continue
                iou = box_iou(track["box"], d["box"])
                sim = cos_sim(track["emb"], d["emb"]) if track["emb"] is not None and d["emb"] is not None else 0
                score = iou + self.lambda_sim * sim if sim > self.sim_thresh else iou
                if score > best_score and iou > self.iou_thresh:
                    best_score = score
                    best_det_idx = i
            if best_det_idx >= 0:
                det = dets[best_det_idx]
                track["box"] = det["box"]
                track["cls"] = det["cls"]
                track["sim"] = det["sim"]
                track["score"] = det["score"]
                track["emb"] = det["emb"]
                track["age"] = 0
                matched.add(best_det_idx)

        for i, d in enumerate(dets):
            if i not in matched:
                self.tracks[self.next_id] = {
                    "box": d["box"],
                    "cls": d["cls"],
                    "sim": d["sim"],
                    "score": d["score"],
                    "emb": d["emb"],
                    "age": 0
                }
                self.next_id += 1

        return [{"id": tid, **track} for tid, track in self.tracks.items()]

class YOLO_SAM2_Detector:
    def __init__(self, yolo_ckpt, sam2_ckpt, sam2_cfg):
        print("[INIT] Loading YOLOv11...")
        self.yolo = YOLO(yolo_ckpt)
        self.yolo.to(device)
        print("[INFO] YOLO loaded.")
        self.cls_color_map = {}

        for cls_id, cls_name in self.yolo.model.names.items():
            random.seed(cls_id * 999)   # cố định màu theo class
            self.cls_color_map[cls_name] = (
                random.randint(60, 255),
                random.randint(60, 255),
                random.randint(60, 255)
            )
        print("[INFO] Auto color map generated for classes:", self.cls_color_map)
        print("[INIT] Loading SAM2...")
        sd, info = load_sam2_weights(sam2_ckpt)
        self.classes = info.get("classes", [])
        self.model = build_sam2(sam2_cfg, None, device)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.float().to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)
        print("[INFO] SAM2 loaded.")

        self.backbone = build_backbone(pretrained=True)
        self.prototypes = build_prototypes(PATHS["ANNOTATED_DIR"], self.classes, self.backbone, max_per_class=80)
        for k in list(self.prototypes.keys()):
            v = self.prototypes[k].ravel()
            self.prototypes[k] = v / (np.linalg.norm(v) + 1e-8)

    def _match_class(self, crop):
        if crop.size == 0:
            return "Unknown", 0.0, None
        inp = preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.backbone(inp).squeeze().cpu().numpy().ravel()
        qn = q / (np.linalg.norm(q) + 1e-8)
        best_cls, best_sc = "Unknown", -1
        for cls, proto in self.prototypes.items():
            sc = float(np.dot(qn, proto))
            if sc > best_sc:
                best_sc, best_cls = sc, cls
        return best_cls, best_sc, qn

    @torch.inference_mode()
    def infer_frame(self, frame, conf_thres=0.35, proto_sim_th=0.40):
        scale_factor = 1.0

        """
        YOLO detect → SAM2 refine mask (batch) → multi-object detection
        """
        detections = []

        yolo_results = self.yolo.predict(frame, conf=conf_thres, verbose=False)

        boxes = []
        cls_ids = []
        scores_yolo = []

        for r in yolo_results:
            if len(r.boxes) == 0:
                continue

            xyxys = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy()

            for b, c, cls_i in zip(xyxys, confs, clses):
                boxes.append(tuple(map(int, b)))
                scores_yolo.append(float(c))
                cls_ids.append(int(cls_i))


        if len(boxes) == 0:
            return []

        if not boxes:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        boxes_np = np.array(boxes, dtype=np.int32)

        try:
            masks, scores, _ = self.predictor.predict(box=boxes_np, multimask_output=True)
        except Exception as e:
            print("[WARN] SAM2 batch inference failed:", e)
            return []

        if isinstance(masks, list):
            all_masks, all_scores = [], []
            for i, mset in enumerate(masks):
                if isinstance(mset, (list, np.ndarray)):
                    for j, m in enumerate(mset):
                        all_masks.append(np.array(m))
                        val = scores[i][j] if isinstance(scores[i], (list, np.ndarray)) else scores[i]
                        all_scores.append(float(val))
            masks = np.array(all_masks)
            scores = np.array(all_scores)

        if masks.ndim == 4:
            masks = masks[:, 0]
        H, W = frame.shape[:2]
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if isinstance(score, (list, np.ndarray)):
                score = float(np.max(score))
            else:
                score = float(score)
            if score < 0.30:
                continue

            mask = masks[i]
            if mask.ndim > 2:
                mask = mask[0]

            mask = mask.astype(np.uint8)

            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            color = self.cls_color_map.get(self.yolo.model.names[cls_ids[i]], (0, 255, 0))

            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[mask > 0] = color

            frame = cv2.addWeighted(frame, 1.0, overlay, 0.45, 0)

            if mask.sum() < 200:
                continue

            coords = np.argwhere(mask > 0)
            if coords.size == 0:
                continue

            ys, xs = coords[:, 0], coords[:, 1]

            bx1 = int(xs.min())
            by1 = int(ys.min())
            bx2 = int(xs.max())
            by2 = int(ys.max())


            bx1 = max(0, bx1)
            by1 = max(0, by1)
            bx2 = min(W-1, bx2)
            by2 = min(H-1, by2)


            x1, y1, x2, y2 = boxes[i]

            detections.append({
                "box": [x1, y1, x2, y2],
                "score": scores_yolo[i],
                "cls": self.yolo.model.names[cls_ids[i]],
                "sim": 1.0,
                "emb": None
            })



        if not detections:
            return []

        boxes_arr = np.array([d["box"] for d in detections], dtype=np.int32)
        scores_arr = np.array([d["score"] * d["sim"] for d in detections], dtype=np.float32)
        keep_idx = nms_boxes(boxes_arr, scores_arr, iou_thresh=0.3)
        final_dets = [detections[i] for i in keep_idx]

        print(f"[INFO] Detected {len(final_dets)} objects.")
        return final_dets



def run_webcam_hybrid():
    print("[INFO] Initialize YOLO + SAM2 hybrid detector...")
    detector = YOLO_SAM2_Detector(
        yolo_ckpt=os.path.join(BASE_DIR, "checkpoints", "yolo11n.pt"),
        sam2_ckpt=PATHS["SAM2_CKPT"],
        sam2_cfg=PATHS["SAM2_CONFIG"]
    )
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INFO] Webcam ready — press 'Q' to exit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        start = time.time()

        dets = detector.infer_frame(frame, conf_thres=0.35, proto_sim_th=0.40)
        tracked = tracker.update(dets)

        out = frame
        total = len(tracked)
        class_counts = {}
        for d in tracked:
            cls = d["cls"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        header_text = f"Detected: {total}  |  " + "  |  ".join([f"{cls}: {cnt}" for cls, cnt in class_counts.items()])
        (hh, ww) = out.shape[:2]

        cv2.rectangle(out, (0, 0), (ww, 40), (30, 30, 30), -1)
        cv2.putText(out, header_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 100), 2)

        cls_color_map = {
            "person": (0, 255, 170),
            "cup": (0, 220, 255),
            "toothbrush": (150, 255, 0),
            "bottle": (255, 180, 0),
            "chair": (255, 120, 200),
            "book": (200, 255, 255),
            "bowl": (180, 140, 255),
            "dining_table": (255, 90, 100),
        }

        for d in tracked:
            x1, y1, x2, y2 = map(int, d["box"])
            cls = d["cls"]
            score = d["score"]

            color = cls_color_map.get(cls, (0, 255, 0))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1),
                        color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

        fps = 1.0 / (time.time() - start + 1e-6)
        fps_text = f"FPS: {fps:.1f}"

        (hh, ww) = out.shape[:2]

        (font_w, font_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)

        pad = 10
        x1 = ww - font_w - 40   # FIXED: dùng ww
        y1 = 50
        x2 = ww - 10
        y2 = y1 + font_h + 20

        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

        alpha = 0.25
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        cv2.putText(out, fps_text, (x1 + 12, y1 + font_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


        cv2.imshow("YOLO + SAM2 Multi-Object Detection", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_hybrid()
