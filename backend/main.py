from flask import Flask, request, jsonify, send_file, make_response, Response
import numpy as np
import cv2
from ultralytics import YOLO
import time
import os
import io
import uuid
import base64
import json
from datetime import datetime
import open3d as o3d
from PIL import Image


# Initialize YOLO model for object detection (if needed)
model = YOLO("yolov8n.pt")

# Define function to check for critical alerts based on removed objects
def check_critical_alerts(room_id, removed_objects):
    """Checks removed objects against stored critical objects for this room"""
    critical_removed = []
    
    try:
        with open("critical_objects.json", "r") as f:
            critical_data = json.load(f)
        
        for obj in removed_objects:
            obj_id = obj.get("removed_object_id")
            if obj_id and any(c["id"] == obj_id and c["isCritical"] for c in critical_data):
                critical_removed.append(obj_id)
    
    except Exception as e:
        print(f"Failed to check critical alerts: {e}")
    
    return critical_removed


class RoomBaselineScanner:
    def __init__(self, storage_path="scan_storage"):
        self.baseline_scans = {}
        self.storage_path = storage_path
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs(os.path.join(storage_path, "rooms"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(storage_path, "pointclouds"), exist_ok=True)

    def detect_objects(self, frame):
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({"bbox": [x1, y1, x2, y2], "centroid": [cx, cy]})
        return detections

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
    
    def compare_detections(self, old_dets, new_dets, iou_thresh=0.5, move_thresh=25):
        added, removed, moved = [], [], []
        matched_old, matched_new = set(), set()
        for i, new_obj in enumerate(new_dets):
            best_score, best_match = 0, -1
            for j, old_obj in enumerate(old_dets):
                iou = self.compute_iou(new_obj["bbox"], old_obj["bbox"])
                dist = np.linalg.norm(np.array(new_obj["centroid"]) - np.array(old_obj["centroid"]))
                score = iou - (dist / 300)
                if score > best_score:
                    best_score = score
                    best_match = j
            if best_score > 0.3:
                matched_old.add(best_match)
                matched_new.add(i)
                dist = np.linalg.norm(np.array(new_obj["centroid"]) - np.array(old_dets[best_match]["centroid"]))
                if dist > move_thresh:
                    moved.append(new_obj)
            else:
                added.append(new_obj)
        for j, old_obj in enumerate(old_dets):
            if j not in matched_old:
                removed.append(old_obj)
        return added, removed, moved

    def draw_differences(self, frame, added, removed, moved):
        for obj in added:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Added", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for obj in removed:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Removed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for obj in moved:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Moved", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return frame
        
    def process_image(self, image_path, room_id=None):
        """
        Process an image to create a baseline scan
        
        Args:
            image_path: Path to the image file or numpy array of the image
            room_id: Identifier for the room (optional)
            
        Returns:
            Dictionary containing processed scan data
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
        else:
            image = image_path
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        # Create corner detection with Harris corner detector
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)
        
        # Threshold for corner detection
        threshold = 0.01 * corners.max()
        corner_points = np.where(corners > threshold)
        corner_coordinates = list(zip(corner_points[1], corner_points[0]))  # x, y format
        
        # Create depth map estimation using gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize gradient magnitude for depth visualization
        depth_map = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        
        # Apply colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        
        # Create room layout estimation
        # Find floor-wall boundaries using line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=100, 
                               minLineLength=100, 
                               maxLineGap=10)
        
        # Store the processed scan
        scan_data = {
            'edges': edges,
            'contours': significant_contours,
            'corners': corner_coordinates,
            'depth_map': depth_map,
            'depth_colored': depth_colored,
            'lines': lines,
            'original_image': image,
            'timestamp': datetime.now().isoformat()
        }
        
        if room_id:
            self.baseline_scans[room_id] = scan_data
            
            # Save scan data to disk for persistence
            self._save_scan_data(room_id, scan_data)
            
        return scan_data
    
    def _save_scan_data(self, room_id, scan_data):
        """Save scan data to disk for persistence"""
        room_dir = os.path.join(self.storage_path, "rooms", room_id)
        os.makedirs(room_dir, exist_ok=True)
        
        # Save original image
        orig_img_path = os.path.join(room_dir, "original.jpg")
        cv2.imwrite(orig_img_path, scan_data['original_image'])
        
        # Save edges
        edges_path = os.path.join(room_dir, "edges.png")
        cv2.imwrite(edges_path, scan_data['edges'])
        
        # Save depth map
        depth_path = os.path.join(room_dir, "depth.png")
        cv2.imwrite(depth_path, scan_data['depth_map'])
        
        # Save depth colored
        depth_colored_path = os.path.join(room_dir, "depth_colored.png")
        cv2.imwrite(depth_colored_path, scan_data['depth_colored'])
        
        # Save metadata (excluding large numpy arrays and opencv objects)
        metadata = {
            'timestamp': scan_data['timestamp'],
            'num_contours': len(scan_data['contours']),
            'num_corners': len(scan_data['corners']),
            'has_lines': scan_data['lines'] is not None
        }
        
        with open(os.path.join(room_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
    
    def _load_scan_data(self, room_id):
        """Load scan data from disk"""
        room_dir = os.path.join(self.storage_path, "rooms", room_id)
        
        if not os.path.exists(room_dir):
            return None
        
        # Load original image
        orig_img_path = os.path.join(room_dir, "original.jpg")
        original_image = cv2.imread(orig_img_path)
        
        # Load edges
        edges_path = os.path.join(room_dir, "edges.png")
        edges = cv2.imread(edges_path, cv2.IMREAD_GRAYSCALE)
        
        # Load depth map
        depth_path = os.path.join(room_dir, "depth.png")
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        
        # Load depth colored
        depth_colored_path = os.path.join(room_dir, "depth_colored.png")
        depth_colored = cv2.imread(depth_colored_path)
        
        # Load metadata
        with open(os.path.join(room_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Re-process the image to get contours, corners and lines
        scan_data = self.process_image(original_image)
        
        # Update with timestamp from metadata
        scan_data['timestamp'] = metadata['timestamp']
        
        return scan_data
    
    def get_all_rooms(self):
        """Get list of all room IDs"""
        rooms_dir = os.path.join(self.storage_path, "rooms")
        if os.path.exists(rooms_dir):
            return [d for d in os.listdir(rooms_dir) 
                   if os.path.isdir(os.path.join(rooms_dir, d))]
        return []
    
    def visualize_scan(self, scan_data, mode='all'):
        """
        Visualize the processed scan data
        
        Args:
            scan_data: Processed scan data dictionary
            mode: Visualization mode ('all', 'edges', 'depth', 'corners', 'contours')
            
        Returns:
            Visualization image or multiple images
        """
        orig_image = scan_data['original_image'].copy()
        edges = scan_data['edges']
        depth_colored = scan_data['depth_colored']
        
        if mode == 'edges':
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif mode == 'depth':
            return depth_colored
        
        elif mode == 'corners':
            corner_image = orig_image.copy()
            for x, y in scan_data['corners']:
                cv2.circle(corner_image, (x, y), 3, (0, 255, 0), -1)
            return corner_image
        
        elif mode == 'contours':
            contour_image = np.zeros_like(orig_image)
            cv2.drawContours(contour_image, scan_data['contours'], -1, (0, 255, 0), 2)
            return contour_image
        
        elif mode == 'lines':
            line_image = orig_image.copy()
            if scan_data['lines'] is not None:
                for line in scan_data['lines']:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return line_image
            
        else:  # 'all' mode
            # Create a composite visualization
            h, w = orig_image.shape[:2]
            
            # Create composite image with different scan types
            composite = np.zeros((h*2, w*2, 3), dtype=np.uint8)
            
            # Original image
            composite[:h, :w] = orig_image
            
            # Edge detection
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            composite[:h, w:w*2] = edges_colored
            
            # Depth map
            composite[h:h*2, :w] = depth_colored
            
            # Corner detection
            corner_image = orig_image.copy()
            for x, y in scan_data['corners']:
                cv2.circle(corner_image, (x, y), 3, (0, 255, 0), -1)
                
            # Draw lines on the corner image
            if scan_data['lines'] is not None:
                for line in scan_data['lines']:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(corner_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            composite[h:h*2, w:w*2] = corner_image
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(composite, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(composite, "Edges", (w+10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(composite, "Depth", (10, h+30), font, 1, (255, 255, 255), 2)
            cv2.putText(composite, "Features", (w+10, h+30), font, 1, (255, 255, 255), 2)
            
            return composite
    
    def compare_scans(self, scan_data1, scan_data2):
        """
        Compare two scans and highlight differences
        
        Args:
            scan_data1: First scan data (baseline)
            scan_data2: Second scan data (current)
            
        Returns:
            Comparison visualization and difference metrics
        """
        # Extract edges from both scans
        edges1 = scan_data1['edges']
        edges2 = scan_data2['edges']
        
        # Calculate structural similarity
        # First, resize if dimensions don't match
        if edges1.shape != edges2.shape:
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        
        # Simple difference
        edge_diff = cv2.absdiff(edges1, edges2)
        
        # Calculate contour differences
        contours1 = scan_data1['contours']
        contours2 = scan_data2['contours']
        
        # Create mask images for contours
        mask1 = np.zeros_like(edges1)
        mask2 = np.zeros_like(edges2)
        
        cv2.drawContours(mask1, contours1, -1, 255, 1)
        cv2.drawContours(mask2, contours2, -1, 255, 1)
        
        contour_diff = cv2.absdiff(mask1, mask2)
        
        # Create a combined difference visualization
        h, w = edges1.shape[:2]
        
        # Create composite image to show differences
        img1 = scan_data1['original_image'].copy()
        img2 = scan_data2['original_image'].copy()
        
        # Resize if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Create difference heatmap
        diff_img = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        diff_heatmap = cv2.applyColorMap(diff_thresh, cv2.COLORMAP_JET)
        
        # Composite image with differences
        composite = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Original baseline
        composite[:h, :w] = img1
        
        # Current scan
        composite[:h, w:w*2] = img2
        
        # Edge differences
        edge_diff_color = cv2.cvtColor(edge_diff, cv2.COLOR_GRAY2BGR)
        edge_diff_color = cv2.applyColorMap(edge_diff, cv2.COLORMAP_HOT)
        composite[h:h*2, :w] = edge_diff_color
        
        # Overall differences
        composite[h:h*2, w:w*2] = diff_heatmap
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(composite, "Baseline", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(composite, "Current", (w+10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(composite, "Edge Diff", (10, h+30), font, 1, (255, 255, 255), 2)
        cv2.putText(composite, "Change Heatmap", (w+10, h+30), font, 1, (255, 255, 255), 2)
        
        # Calculate difference metrics
        total_diff = np.sum(diff_gray) / (255 * diff_gray.size)
        edge_diff_metric = np.sum(edge_diff) / (255 * edge_diff.size)
        
        diff_metrics = {
            'total_difference_percent': float(total_diff * 100),
            'edge_difference_percent': float(edge_diff_metric * 100),
            'significant_changes': bool(total_diff > 0.05)  # Threshold for significant changes
        }
        
        return composite, diff_metrics
    
    def generate_3d_pointcloud(self, scan_data):
        """
        Generate a 3D point cloud from depth map
        
        Args:
            scan_data: Processed scan data
            
        Returns:
            Open3D point cloud object
        """
        # Use depth map to create point cloud
        depth = scan_data['depth_map'].astype(np.float32)
        
        # Create coordinate grid
        h, w = depth.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Scale depth for better visualization
        depth_scaled = depth / 255.0 * 5.0  # Scale to 0-5 meters for visualization
        
        # Create points
        points = np.zeros((h*w, 3))
        points[:, 0] = x.flatten()
        points[:, 1] = y.flatten()
        points[:, 2] = depth_scaled.flatten()
        
        # Create colors
        rgb_image = cv2.cvtColor(scan_data['original_image'], cv2.COLOR_BGR2RGB)
        colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to 0-1
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def save_pointcloud(self, pcd, room_id):
        """Save point cloud to PLY file for web viewing"""
        filename = os.path.join(self.storage_path, "pointclouds", f"{room_id}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        return filename
    
    def image_to_base64(self, img):
        """Convert OpenCV image to base64 string for web viewing"""
        _, buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return img_str
    
    def save_visualization(self, img, filename):
        """Save visualization to disk"""
        vis_path = os.path.join(self.storage_path, "visualizations", filename)
        cv2.imwrite(vis_path, img)
        return vis_path

# Create Flask API
app = Flask(__name__)
scanner = RoomBaselineScanner()
stream_active = False

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE')
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/live-feed', methods=['GET'])
def live_feed():
    global stream_active
    stream_active = True 
    def generate():
        global stream_active
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        prev_detections = []
        
        try:
            while stream_active:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = scanner.detect_objects(frame)
                added, removed, moved = scanner.compare_detections(prev_detections, detections)
                frame = scanner.draw_differences(frame, added, removed, moved)

                msg = f"Added: {len(added)} | Removed: {len(removed)} | Moved: {len(moved)}"
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    break

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                prev_detections = detections
                time.sleep(0.03)

        except GeneratorExit:
            print("Client disconnected.")
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            cap.release()
            print("Camera released.")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stop-stream', methods=['POST'])
def stop_stream():
    global stream_active
    stream_active = False
    return jsonify({'status': 'stopped', 'message': 'Live stream has been stopped.'})

@app.route('/api/stream-status', methods=['GET'])
def stream_status():
    return jsonify({'stream_active': stream_active})


@app.route('/api/rooms', methods=['GET'])
def get_rooms():
    """Get all rooms that have baseline scans"""
    rooms = scanner.get_all_rooms()
    
    # Get metadata for each room
    room_data = []
    for room_id in rooms:
        room_dir = os.path.join(scanner.storage_path, "rooms", room_id)
        metadata_path = os.path.join(room_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            room_data.append({
                'room_id': room_id,
                'timestamp': metadata['timestamp'],
                'thumbnail': f"/api/rooms/{room_id}/original"
            })
    
    return jsonify({
        'status': 'success',
        'rooms': room_data
    })

@app.route('/api/rooms/<room_id>', methods=['GET'])
def get_room_info(room_id):
    """Get room details"""
    room_dir = os.path.join(scanner.storage_path, "rooms", room_id)
    
    if not os.path.exists(room_dir):
        return jsonify({'error': f'Room {room_id} not found'}), 404
    
    metadata_path = os.path.join(room_dir, "metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return jsonify({
        'status': 'success',
        'room_id': room_id,
        'metadata': metadata,
        'scans': {
            'original': f"/api/rooms/{room_id}/original",
            'edges': f"/api/rooms/{room_id}/edges",
            'depth': f"/api/rooms/{room_id}/depth",
            'depth_colored': f"/api/rooms/{room_id}/depth_colored",
            'corners': f"/api/rooms/{room_id}/corners",
            'lines': f"/api/rooms/{room_id}/lines",
            'all': f"/api/rooms/{room_id}/all"
        }
    })

@app.route('/api/rooms/<room_id>/original', methods=['GET'])
def get_room_original(room_id):
    """Get original image for room"""
    room_dir = os.path.join(scanner.storage_path, "rooms", room_id)
    orig_img_path = os.path.join(room_dir, "original.jpg")
    
    if not os.path.exists(orig_img_path):
        return jsonify({'error': f'Original image for room {room_id} not found'}), 404
    
    return send_file(orig_img_path, mimetype='image/jpeg')

@app.route('/api/rooms/<room_id>/<scan_type>', methods=['GET'])
def get_room_scan(room_id, scan_type):
    """Get specific scan visualization for room"""
    valid_types = ['edges', 'depth', 'depth_colored', 'corners', 'lines', 'all', 'contours']
    
    if scan_type not in valid_types:
        return jsonify({'error': f'Invalid scan type. Must be one of {valid_types}'}), 400
    
    # Check if this is a direct file we can serve
    if scan_type == 'edges' or scan_type == 'depth' or scan_type == 'depth_colored':
        room_dir = os.path.join(scanner.storage_path, "rooms", room_id)
        file_path = os.path.join(room_dir, f"{scan_type}.png")
        
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
    
    # Otherwise, we need to load scan data and visualize on-the-fly
    room_scan_data = scanner._load_scan_data(room_id)
    
    if room_scan_data is None:
        return jsonify({'error': f'Room {room_id} not found'}), 404
    
    visualization = scanner.visualize_scan(room_scan_data, mode=scan_type)
    
    # Convert to image bytes
    buffer = io.BytesIO()
    img_pil = Image.fromarray(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png')

@app.route('/api/scan', methods=['POST'])
def create_room_scan():
    """Create a new room scan from an image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Generate room ID if not provided
    room_id = request.form.get('room_id', str(uuid.uuid4()))
    
    # Read image
    img_array = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # Process image
    try:
        scan_data = scanner.process_image(image, room_id)
        
        # Generate visualizations for immediate use
        all_vis = scanner.visualize_scan(scan_data, mode='all')
        edges_vis = scanner.visualize_scan(scan_data, mode='edges')
        depth_vis = scanner.visualize_scan(scan_data, mode='depth')
        corners_vis = scanner.visualize_scan(scan_data, mode='corners')
        
        # Generate and save point cloud
        pcd = scanner.generate_3d_pointcloud(scan_data)
        scanner.save_pointcloud(pcd, room_id)
        
        # Save visualizations for future use
        scanner.save_visualization(all_vis, f"{room_id}_all.png")
        
        # Create response with image URLs
        # Get object detection results from the original image
        results = model(image)[0]
        boxes = []
        
        # Format detection results for frontend
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = results.names[class_id]
            
            boxes.append({
                'id': f"{room_id}_obj_{i}",  # Unique ID for each detected object
                'label': f"{label} ({confidence:.2f})",
                'bbox': [x1, y1, x2, y2]
            })
            room_dir = os.path.join(scanner.storage_path, "rooms", room_id)
            os.makedirs(room_dir, exist_ok=True)
            # Save boxes with IDs to JSON file
            with open(os.path.join(room_dir, "original_boxes.json"), "w") as f:
                json.dump(boxes, f)
        
        response = {
            'status': 'success',
            'room_id': room_id,
            'message': 'Room scan created successfully',
            'timestamp': scan_data['timestamp'],
            'boxes': boxes,
            'original_image_url': f"/api/rooms/{room_id}/original",
            'scans': {
            'original': f"/api/rooms/{room_id}/original",
            'edges': f"/api/rooms/{room_id}/edges",
            'depth': f"/api/rooms/{room_id}/depth",
            'depth_colored': f"/api/rooms/{room_id}/depth_colored",
            'corners': f"/api/rooms/{room_id}/corners",
            'lines': f"/api/rooms/{room_id}/lines",
            'all': f"/api/rooms/{room_id}/all"
            }
        }
        
        # Include base64 encoded images in response if requested
        include_images = request.form.get('include_images', 'false').lower() == 'true'
        if include_images:
            response['images'] = {
                'all': scanner.image_to_base64(all_vis),
                'edges': scanner.image_to_base64(edges_vis),
                'depth': scanner.image_to_base64(depth_vis),
                'corners': scanner.image_to_base64(corners_vis)
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_room_scans():
    """Compare a new scan with a baseline scan"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    room_id = request.form.get('room_id')
    if not room_id:
        return jsonify({'error': 'No room_id provided for comparison'}), 400
    
    # Check if baseline scan exists
    if not os.path.exists(os.path.join(scanner.storage_path, "rooms", room_id)):
        return jsonify({'error': f'No baseline scan found for room {room_id}'}), 404
    
    try:
        # Load baseline scan
        baseline_scan = scanner._load_scan_data(room_id)
        
        # Read uploaded image
        img_array = np.frombuffer(image_file.read(), np.uint8)
        current_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if current_image is None:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Process current image
        current_scan = scanner.process_image(current_image)
        
        # Generate comparison visualization
        comparison_img, diff_metrics = scanner.compare_scans(baseline_scan, current_scan)
        comparison_id = f"{room_id}_{int(time.time())}"
        scanner.save_visualization(comparison_img, f"{comparison_id}_comparison.png")
        
        # Create response with comparison results
        response = {
            'status': 'success',
            'room_id': room_id,
            'comparison_id': comparison_id,
            'comparison_url': f"/api/comparisons/{comparison_id}",
            'metrics': diff_metrics
        }
        
        # Include base64 encoded comparison in response if requested
        include_images = request.form.get('include_images', 'false').lower() == 'true'
        if include_images:
            response['comparison_image'] = scanner.image_to_base64(comparison_img)
        
        # Detect objects in both images
        base_dets = scanner.detect_objects(baseline_scan['original_image'])
        curr_dets = scanner.detect_objects(current_image)
        added, removed, moved = scanner.compare_detections(base_dets, curr_dets)

        # Assign consistent removed object IDs
        removed_with_ids = []
        for i, obj in enumerate(removed):
            bbox = obj.get("bbox")
            obj_with_details = obj.copy()  # Start with existing properties
            
            # Try to match with original boxes to get complete details
            if bbox:
                room_boxes_path = os.path.join(scanner.storage_path, "rooms", room_id, "original_boxes.json")
                if os.path.exists(room_boxes_path):
                    with open(room_boxes_path, "r") as f:
                        room_boxes = json.load(f)
                        #print("Room Boxes:", room_boxes)
                        for b in room_boxes:
                            print("Checking box:", b)
                            print("Against bbox:", bbox)
                            if b["bbox"] == bbox:
                                print("Matching box found:", b)
                                # Add all details from the original box
                                obj_with_details["removed_object_id"] = b["id"]
                                obj_with_details["label"] = b["label"]
                                obj_with_details["bbox"] = b["bbox"]
                                break
            
            removed_with_ids.append(obj_with_details)

        # Add object changes to response
        response["changes"] = {
            "added": added,
            "removed": removed_with_ids,
            "moved": moved
        }
        
        # Check for critical alerts
        critical_removed = check_critical_alerts(room_id, removed_with_ids)
        response["critical_alerts"] = critical_removed
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/comparisons/<comparison_id>', methods=['GET'])
def get_comparison(comparison_id):
    """Get a previously generated comparison visualization"""
    comparison_path = os.path.join(scanner.storage_path, "visualizations", f"{comparison_id}_comparison.png")
    
    if not os.path.exists(comparison_path):
        return jsonify({'error': f'Comparison {comparison_id} not found'}), 404
    
    return send_file(comparison_path, mimetype='image/png')

@app.route('/api/pointclouds/<room_id>', methods=['GET'])
def get_pointcloud(room_id):
    """Get a 3D point cloud file for a room"""
    pointcloud_path = os.path.join(scanner.storage_path, "pointclouds", f"{room_id}.ply")
    
    if not os.path.exists(pointcloud_path):
        # Generate on-the-fly if not found
        room_scan_data = scanner._load_scan_data(room_id)
        
        if room_scan_data is None:
            return jsonify({'error': f'Room {room_id} not found'}), 404
        
        pcd = scanner.generate_3d_pointcloud(room_scan_data)
        pointcloud_path = scanner.save_pointcloud(pcd, room_id)
    
    return send_file(pointcloud_path, 
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name=f"{room_id}_pointcloud.ply")

# Serve documentation via Swagger UI
@app.route('/api/docs', methods=['GET'])
def get_docs():
    """Get API documentation"""
    # Simple HTML page with Swagger UI
    swagger_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Room Scanner API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.1.0/swagger-ui.min.css" />
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.1.0/swagger-ui-bundle.min.js"></script>
         <script>
          window.onload = function() {
            const ui = SwaggerUIBundle({
              url: "/api/swagger.json",
              dom_id: '#swagger-ui',
              presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.SwaggerUIStandalonePreset
              ],
              layout: "StandaloneLayout"
            });
          }
        </script>
      </body>
    </html>
    """
    return swagger_html

@app.route('/api/swagger.json', methods=['GET'])
def swagger_json():
    """Serve Swagger specification"""
    swagger_spec = {
        "swagger": "2.0",
        "info": {
            "title": "Room Scanner API",
            "version": "1.0.0",
            "description": "API for processing room scans, generating visualizations, and comparing scans."
        },
        "host": "localhost:5000",
        "basePath": "/api",
        "schemes": [
            "http"
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "responses": {
                        "200": {
                            "description": "Server is running"
                        }
                    }
                }
            },
            "/rooms": {
                "get": {
                    "summary": "Get all rooms with baseline scans",
                    "responses": {
                        "200": {
                            "description": "A list of room IDs and metadata"
                        }
                    }
                }
            },
            "/rooms/{room_id}": {
                "get": {
                    "summary": "Get room details",
                    "parameters": [
                        {
                            "name": "room_id",
                            "in": "path",
                            "required": True,
                            "type": "string",
                            "description": "Unique room identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Room details and scan endpoints"
                        },
                        "404": {
                            "description": "Room not found"
                        }
                    }
                }
            },
            "/rooms/{room_id}/{scan_type}": {
                "get": {
                    "summary": "Get specific scan visualization for room",
                    "parameters": [
                        {
                            "name": "room_id",
                            "in": "path",
                            "required": True,
                            "type": "string",
                            "description": "Unique room identifier"
                        },
                        {
                            "name": "scan_type",
                            "in": "path",
                            "required": True,
                            "type": "string",
                            "enum": ["edges", "depth", "depth_colored", "corners", "lines", "all", "contours"],
                            "description": "Type of scan visualization"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Scan visualization image"
                        },
                        "400": {
                            "description": "Invalid scan type"
                        },
                        "404": {
                            "description": "Room or scan not found"
                        }
                    }
                }
            },
            "/scan": {
                "post": {
                    "summary": "Create a new room scan from an image",
                    "consumes": [
                        "multipart/form-data"
                    ],
                    "parameters": [
                        {
                            "name": "image",
                            "in": "formData",
                            "required": True,
                            "type": "file",
                            "description": "Image file to process"
                        },
                        {
                            "name": "room_id",
                            "in": "formData",
                            "required": False,
                            "type": "string",
                            "description": "Optional room identifier"
                        },
                        {
                            "name": "include_images",
                            "in": "formData",
                            "required": False,
                            "type": "string",
                            "description": "Whether to include base64 encoded images in the response (true/false)"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Room scan created successfully"
                        },
                        "400": {
                            "description": "Invalid input"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/compare": {
                "post": {
                    "summary": "Compare a new scan with an existing baseline scan",
                    "consumes": [
                        "multipart/form-data"
                    ],
                    "parameters": [
                        {
                            "name": "image",
                            "in": "formData",
                            "required": True,
                            "type": "file",
                            "description": "Current image file to compare"
                        },
                        {
                            "name": "room_id",
                            "in": "formData",
                            "required": True,
                            "type": "string",
                            "description": "Room identifier for comparison"
                        },
                        {
                            "name": "include_images",
                            "in": "formData",
                            "required": False,
                            "type": "string",
                            "description": "Whether to include base64 encoded comparison image in the response (true/false)"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Comparison results with metrics and visualization"
                        },
                        "400": {
                            "description": "Invalid input"
                        },
                        "404": {
                            "description": "Baseline scan not found"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/comparisons/{comparison_id}": {
                "get": {
                    "summary": "Get a previously generated comparison visualization",
                    "parameters": [
                        {
                            "name": "comparison_id",
                            "in": "path",
                            "required": True,
                            "type": "string",
                            "description": "Unique comparison identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Comparison visualization image"
                        },
                        "404": {
                            "description": "Comparison not found"
                        }
                    }
                }
            },
            "/pointclouds/{room_id}": {
                "get": {
                    "summary": "Get a 3D point cloud file for a room",
                    "parameters": [
                        {
                            "name": "room_id",
                            "in": "path",
                            "required": True,
                            "type": "string",
                            "description": "Unique room identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "3D point cloud file (PLY format)"
                        },
                        "404": {
                            "description": "Room not found"
                        }
                    }
                }
            }
        }
    }
    return jsonify(swagger_spec)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)