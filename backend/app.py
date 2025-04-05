from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import open_clip
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.optimize import linear_sum_assignment
import os
import base64
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load models
model = YOLO('yolov8l.pt')
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Utility Functions
def get_clip_embedding(image):
    image = Image.fromarray(image)
    image = clip_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def normalize_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def align_images(img1, img2):
    gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return img2
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(des1, des2)
    if len(matches) < 4:
        return img2
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0])) if H is not None else img2

def detect_objects(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.55:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        clip_feat = get_clip_embedding(crop)
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "centroid": [(x1 + x2) // 2, (y1 + y2) // 2],
            "class_id": cls_id,
            "confidence": conf,
            "clip_embedding": clip_feat
        })
    return detections

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)

def match_detections(old_dets, new_dets):
    if not old_dets or not new_dets:
        return new_dets, old_dets, []
    cost_matrix = np.zeros((len(old_dets), len(new_dets)))
    for i, old in enumerate(old_dets):
        for j, new in enumerate(new_dets):
            sim = 0.6 * cosine_similarity(old["clip_embedding"], new["clip_embedding"]) + \
                  0.4 * compute_iou(old["bbox"], new["bbox"])
            cost_matrix[i][j] = 1 - sim
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    added, removed, moved = [], [], []
    matched_old, matched_new = set(), set()
    for i, j in zip(row_ind, col_ind):
        sim = 1 - cost_matrix[i][j]
        if sim < 0.25:
            continue
        old, new = old_dets[i], new_dets[j]
        dist = np.linalg.norm(np.array(old["centroid"]) - np.array(new["centroid"]))
        if dist > 15:
            new["prev_centroid"] = old["centroid"]
            moved.append(new)
        matched_old.add(i)
        matched_new.add(j)
    removed = [o for i, o in enumerate(old_dets) if i not in matched_old]
    added = [n for j, n in enumerate(new_dets) if j not in matched_new]
    return added, removed, moved

def draw_differences(image, added, removed, moved):
    img = image.copy()
    for obj in removed:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, 'Removed', (x1, y1 - 10), 0, 0.6, (0, 0, 255), 2)
    for obj in added:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, 'Added', (x1, y1 - 10), 0, 0.6, (0, 255, 0), 2)
    for obj in moved:
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["centroid"]
        pcx, pcy = obj["prev_centroid"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(img, 'Moved', (x1, y1 - 10), 0, 0.6, (0, 165, 255), 2)
        cv2.arrowedLine(img, (pcx, pcy), (cx, cy), (0, 165, 255), 2)
    return img

@app.route("/compare", methods=["POST"])
def compare_images():
    if "base" not in request.files or "current" not in request.files:
        return jsonify({"error": "Both 'base' and 'current' image files are required"}), 400

    base_img = cv2.imdecode(np.frombuffer(request.files["base"].read(), np.uint8), 1)
    curr_img = cv2.imdecode(np.frombuffer(request.files["current"].read(), np.uint8), 1)

    base_img, curr_img = normalize_lighting(base_img), normalize_lighting(curr_img)
    curr_img = align_images(base_img, curr_img)

    base_dets = detect_objects(base_img)
    curr_dets = detect_objects(curr_img)

    added, removed, moved = match_detections(base_dets, curr_dets)

    annotated = draw_differences(curr_img, added, removed, moved)
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    base64_url = f"data:image/jpeg;base64,{encoded_image}"

    stats = {
        "total_in_base": len(base_dets),
        "total_in_current": len(curr_dets),
        "total_difference": abs(len(base_dets) - len(curr_dets)),
        "total_added": len(added),
        "total_removed": len(removed),
        "total_moved": len(moved),
        "increase_in_objects": len(curr_dets) - len(base_dets),
        "decrease_in_objects": len(base_dets) - len(curr_dets),
    }

    return jsonify({
        "image_base64": base64_url,
        "stats": stats
    })

@app.route("/compare/json", methods=["POST"])
def compare_json_only():
    if "base" not in request.files or "current" not in request.files:
        return jsonify({"error": "Both 'base' and 'current' image files are required"}), 400

    base_img = cv2.imdecode(np.frombuffer(request.files["base"].read(), np.uint8), 1)
    curr_img = cv2.imdecode(np.frombuffer(request.files["current"].read(), np.uint8), 1)

    base_img, curr_img = normalize_lighting(base_img), normalize_lighting(curr_img)
    curr_img = align_images(base_img, curr_img)

    base_dets = detect_objects(base_img)
    curr_dets = detect_objects(curr_img)

    added, removed, moved = match_detections(base_dets, curr_dets)

    stats = {
        "total_in_base": len(base_dets),
        "total_in_current": len(curr_dets),
        "total_difference": abs(len(base_dets) - len(curr_dets)),
        "total_added": len(added),
        "total_removed": len(removed),
        "total_moved": len(moved),
        "increase_in_objects": len(curr_dets) - len(base_dets),
        "decrease_in_objects": len(base_dets) - len(curr_dets),
        "added_objects": added,
        "removed_objects": removed,
        "moved_objects": moved
    }

    return jsonify(stats)
    # Enable CORS for all routes


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)