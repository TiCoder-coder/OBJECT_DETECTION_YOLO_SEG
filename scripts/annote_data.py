import os, sys, json, cv2, torch, yaml, gc
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from datetime import datetime
from roboflow import Roboflow
import supervision as sv


RF_API_KEY = "utF5EVgFHoqC0xPQRuz4"
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic_light","fire_hydrant","stop_sign","parking_meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports_ball",
    "kite","baseball_bat","baseball_glove","skateboard","surfboard","tennis_racket",
    "bottle","wine_glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","brocolli","carrot","hot_dog","pizza","donut","cake","chair",
    "couch","potted_plant","bed","dining_table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell_phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy_bear","hair_drier","toothbrush"
]


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from sam2.sam2.modeling.sam2_base import SAM2Base
    from sam2.training.trainer import Trainer as SAMTrainer
    from sam2.training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
    from automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2.build_sam import build_sam2
    print("[SUCCESS] SAM2 imported OK.")
except Exception as e:
    print("[ERROR] Import failed:", e)
    raise

CHECKPOINT_YOLO = os.path.join(PROJECT_ROOT, "checkpoints", "yolo11m.pt")
CHECKPOINT_SAM = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_base_plus.pt")
CONFIG_SAM = os.path.join(PROJECT_ROOT, "configs", "sam2.1", "sam2.1_hiera_b+.yaml")

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "resize")

HYBRID_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hybrid_data_test")
YOLO_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hybrid_data_yolo")
os.makedirs(HYBRID_DATA_DIR, exist_ok=True)
os.makedirs(YOLO_DATA_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] Using CPU (annotation will be slow)")

print(f"[INFO] Loading YOLOv11 from: {CHECKPOINT_YOLO}")
yolo = YOLO(CHECKPOINT_YOLO)

print(f"[INFO] Loading SAM2.1 from: {CHECKPOINT_SAM}")
sam_model = build_sam2(CONFIG_SAM, CHECKPOINT_SAM, device=DEVICE)
mask_generator = SAM2AutomaticMaskGenerator(
    sam_model,
    points_per_side=32,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)
print("[READY] Both YOLOv11 and SAM2.1 loaded successfully.")

images, annotations, categories = [], [], {}
image_id, ann_id = 1, 1

print(f"[START] Scanning folder: {RAW_DIR}")

print("loading Roboflow workspace...")

rf = Roboflow(api_key=RF_API_KEY)

workspace = rf.workspace()  
print("Workspace URL:", workspace.url)

try:
    project = workspace.create_project(
        project_name="auto-annotation-yolo-sam",
        project_type="instance-segmentation",
        project_license="MIT",
        annotation="polygon"
    )
    print("[ROBOFLOW] Project created.")
except Exception as e:
    print("[ROBOFLOW] Project already exists, loading existing one...")

    project = workspace.project("auto-annotation-yolo-sam")



print("[ROBOFLOW] Project created:", project)

def nms_numpy(boxes, iou_thresh=0.5):

    if len(boxes) == 0:
        return boxes
    boxes = boxes[np.argsort(boxes[:, 3])]
    keep = []
    while len(boxes) > 0:
        box = boxes[-1]
        keep.append(box)
        boxes = boxes[:-1]
        if len(boxes) == 0:
            break
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        boxes = boxes[iou < iou_thresh]
    return np.array(keep)

for root, _, files in os.walk(RAW_DIR):

    for file_name in tqdm(files, desc="Processing images"):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(root, file_name)
        img = cv2.imread(file_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        results = yolo(img, conf=0.4, device=DEVICE)
        raw_boxes = results[0].boxes
        if not len(raw_boxes):
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue

        has_valid_box = False
        overlay = img.copy()

        xyxy = raw_boxes.xyxy.cpu().numpy()
        conf = raw_boxes.conf.cpu().numpy()
        cls = raw_boxes.cls.cpu().numpy()

        dets = []
        for i in range(len(xyxy)):
            dets.append({
                "xyxy": xyxy[i],
                "conf": float(conf[i]),
                "cls": int(cls[i])
            })

        dets = sorted(dets, key=lambda x: x["conf"], reverse=True)

        filtered = []
        while len(dets) > 0:
            best = dets[0]
            filtered.append(best)

            remains = []
            for other in dets[1:]:
                boxA = best["xyxy"]
                boxB = other["xyxy"]

                xx1 = max(boxA[0], boxB[0])
                yy1 = max(boxA[1], boxB[1])
                xx2 = min(boxA[2], boxB[2])
                yy2 = min(boxA[3], boxB[3])

                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = inter / (areaA + areaB - inter + 1e-6)

                if iou < 0.5:
                    remains.append(other)

            dets = remains


        for det in filtered:
            x1, y1, x2, y2 = map(int, det["xyxy"])
            conf = det["conf"]
            cls_id = det["cls"]

            label = CLASSES[cls_id]

            if label not in categories:
                categories[label] = len(categories) + 1

            pad = 15
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            with torch.no_grad():
                masks = mask_generator.generate(crop)
            if not masks:
                continue

            best_mask = max(
                masks,
                key=lambda m: np.sum(np.array(m["segmentation"], dtype=np.uint8))
            )
            segmentation = np.array(best_mask["segmentation"], dtype=np.uint8)

            seg_coords = np.column_stack(np.where(segmentation > 0))
            seg_coords[:, [0,1]] = seg_coords[:, [1,0]]
            seg_coords[:, 0] += x1p
            seg_coords[:, 1] += y1p

            segmentation_flat = seg_coords.flatten().tolist()

            bbox = [x1, y1, x2 - x1, y2 - y1]

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": categories[label],
                "segmentation": [segmentation_flat],
                "bbox": bbox,
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
                "confidence": conf
            })
            ann_id += 1
            has_valid_box = True


            color = (0,255,0)
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
            cv2.putText(overlay, f"{label} ({conf:.2f})",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        if has_valid_box:
            label_dir = HYBRID_DATA_DIR
            viz_name = f"{os.path.splitext(file_name)[0]}_viz.jpg"
            save_path = os.path.join(label_dir, viz_name)

            cv2.imwrite(save_path, overlay)
            rel_path = os.path.relpath(save_path, HYBRID_DATA_DIR)
            images.append({
                "id": image_id,
                "file_name": rel_path,
                "width": w,
                "height": h
            })
            image_id += 1
        gc.collect()


COCO_PATH = os.path.join(HYBRID_DATA_DIR, "hybrid_coco.json")
LABEL_PATH = os.path.join(HYBRID_DATA_DIR, "label.json")

coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": cid, "name": name} for name, cid in categories.items()]
}

with open(COCO_PATH, "w") as f:
    json.dump(coco_output, f, indent=4)
with open(LABEL_PATH, "w") as f:
    json.dump(categories, f, indent=4)

print(f"\n[SUCCESS] Hybrid annotation saved successfully!")
print(f" → COCO JSON: {COCO_PATH}")
print(f" → Label map: {LABEL_PATH}")
print(f" → Classes ({len(categories)}): {list(categories.keys())}")
print(f" → Total images: {len(images)} | Total annotations: {len(annotations)}")

print("Done — annotation process completed successfully.")
print("[ROBOFLOW] Uploading dataset...")

valid_images = [
    os.path.join(HYBRID_DATA_DIR, f)
    for f in os.listdir(HYBRID_DATA_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]


# project.upload_dataset(
#     image_dir=HYBRID_DATA_DIR,
#     annotation_path=COCO_PATH,
#     annotation_format="coco"
# )



print("[ROBOFLOW] Dataset upload complete.")
