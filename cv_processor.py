import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json
from pathlib import Path
from database import Database
from energy_analyzer import EnergyAnalyzer

class CVProcessor:
    def __init__(self, use_database=True, room_id="CS_LAB_101", verify_location=True, optimization_mode='balanced', model_path=None):
        # Load YOLO model (lightweight nano version for speed)
        # If a local model path is provided, use it to avoid re-downloading
        self.model_path = model_path
        self.model = YOLO(model_path if model_path else 'yolov8n.pt')  # Auto-downloads first time if needed
        self.room_id = room_id  # e.g., "CS_LAB_101", "IT_CLASS_202"
        self.department = room_id.split('_')[0] if '_' in room_id else 'GENERAL'  # Extract dept
        self.verify_location = verify_location  # Enable location verification
        self.location_verified = False  # Track if location was verified via QR code
        self.location_confidence = 0.5  # Default confidence (increases with QR verification)
        
        # Optimization mode: 'precision' (fewer FP), 'recall' (fewer FN), 'balanced' (best F1)
        self.optimization_mode = optimization_mode
        self.set_thresholds_by_mode(optimization_mode)
        
        # QR code detector for room verification
        self.qr_detector = cv2.QRCodeDetector()
        
        # Face detection setup - Multi-scale for better accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Person tracking - ENHANCED ACCURACY
        self.known_faces = {}  # Store: {person_id: {'histograms': [], 'last_bbox': [], 'frame_last_seen': 0, 'detection_count': 0, 'confidence_history': []}}
        self.person_counter = 0  # Counter for assigning new person IDs
        self.person_logs = {}  # Track activities per person: {person_id: [events]}
        self.current_frame_number = 0
        self.person_temporal_buffer = {}  # Buffer for temporal smoothing: {person_id: [recent_detections]}
        self.min_detections_for_verification = 5  # Require 5 consistent detections before high confidence
        
        # Kalman filters for person tracking (reduces false negatives)
        self.kalman_filters = {}  # {person_id: KalmanFilter}
        
        # Background subtraction for improved occupancy detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.occupancy_buffer = []  # Temporal smoothing for occupancy
        self.occupancy_buffer_size = 5  # 5-frame consensus
        
        # Performance metrics tracking
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        # Face database directory
        self.face_db_path = Path('outputs/face_database')
        self.face_db_path.mkdir(parents=True, exist_ok=True)
        
        # Database integration
        self.use_database = use_database
        self.db = Database() if use_database else None
        
        # Energy analyzer for campus sustainability tracking (pass optimization mode)
        self.energy_analyzer = EnergyAnalyzer(self.room_id, optimization_mode=optimization_mode)
        self.previous_devices_state = []  # Track previous device states
        self.previous_occupancy = False
        self.previous_lights_on = False
    
    def set_thresholds_by_mode(self, mode):
        """
        Set detection thresholds optimized for different metrics
        Based on ROC curve analysis for optimal precision/recall trade-off
        """
        if mode == 'precision':  # Minimize false positives
            self.yolo_conf_threshold = 0.35  # Higher confidence (default: 0.25)
            self.person_match_threshold = 0.60  # Higher matching threshold
            self.device_validation_frames = 5  # More frames required
            self.action_confidence_min = 0.80  # Only high-confidence actions
            self.min_detections_for_verification = 8  # More detections needed
            
        elif mode == 'recall':  # Minimize false negatives
            self.yolo_conf_threshold = 0.15  # Lower confidence (catch more)
            self.person_match_threshold = 0.40  # Lower matching threshold
            self.device_validation_frames = 2  # Fewer frames required
            self.action_confidence_min = 0.60  # Accept lower confidence
            self.min_detections_for_verification = 3  # Fewer detections needed
            
        else:  # 'balanced' - Optimal F1 score
            self.yolo_conf_threshold = 0.25  # Balanced confidence
            self.person_match_threshold = 0.50  # Balanced matching
            self.device_validation_frames = 3  # Balanced validation
            self.action_confidence_min = 0.70  # Balanced action threshold
            self.min_detections_for_verification = 5  # Balanced verification
    
    def create_kalman_filter(self, initial_bbox):
        """
        Create Kalman filter for person tracking (reduces false negatives)
        State: [x, y, width, height, dx, dy]
        """
        kalman = cv2.KalmanFilter(6, 4)  # 6 state vars, 4 measurement vars
        
        # State transition matrix (constant velocity model)
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Initialize state
        x1, y1, x2, y2 = initial_bbox
        kalman.statePost = np.array([x1, y1, x2-x1, y2-y1, 0, 0], dtype=np.float32)
        
        return kalman
        
    def verify_room_location(self, frame):
        """
        Verify room location using QR code detection
        Returns: dict with verification status and confidence
        """
        try:
            # Detect QR code in frame
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            
            if data and data.strip():
                # QR code format expected: "ROOM:CS_LAB_101"
                if data.startswith('ROOM:'):
                    detected_room = data.split(':', 1)[1]
                    if detected_room == self.room_id:
                        self.location_verified = True
                        self.location_confidence = 1.0
                        return {'verified': True, 'confidence': 1.0, 'method': 'qr_code', 'room_id': detected_room}
                    else:
                        return {'verified': False, 'confidence': 0.2, 'method': 'qr_mismatch', 'detected_room': detected_room, 'expected_room': self.room_id}
        except Exception:
            pass
        
        return {'verified': self.location_verified, 'confidence': self.location_confidence, 'method': 'default'}
    
    def process_frame(self, frame):
        """
        Detect objects in a single frame
        Returns: detection results
        """
        # Verify location periodically (every 150 frames = 5 seconds)
        if self.verify_location and self.current_frame_number % 150 == 0:
            self.verify_room_location(frame)
        
        results = self.model(frame, verbose=False)
        return results[0]  # First result
    
    def detect_and_recognize_faces(self, frame, person_boxes):
        """
        Detect and recognize faces in person bounding boxes
        Returns: list of person IDs and face locations
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recognized_persons = []
        
        for box in person_boxes:
            x1, y1, x2, y2 = box
            
            # Extract person region
            person_roi = gray[y1:y2, x1:x2]
            person_roi_color = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Multi-pass face detection for higher accuracy
            # Pass 1: Frontal face detection
            faces = self.face_cascade.detectMultiScale(
                person_roi,
                scaleFactor=1.05,  # More sensitive
                minNeighbors=3,    # Lower threshold
                minSize=(20, 20)   # Smaller minimum size
            )
            
            # Pass 2: If no frontal face, try profile detection
            if len(faces) == 0:
                faces = self.face_cascade_profile.detectMultiScale(
                    person_roi,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20)
                )
            
            # If still no face detected, use appearance-based recognition
            if len(faces) == 0:
                # Use the entire person bounding box as "face" for recognition
                face_roi = person_roi
                person_id = self._match_or_create_person_by_appearance(person_roi_color, [x1, y1, x2, y2])
                
                recognized_persons.append({
                    'person_id': person_id,
                    'bbox': [x1, y1, x2, y2],
                    'face_bbox': None,
                    'confidence': 0.5,  # Lower confidence when face not detected
                    'detection_method': 'appearance'
                })
            else:
                # Take the largest face
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                
                # Extract face region
                face_roi = person_roi[fy:fy+fh, fx:fx+fw]
                face_roi_color = person_roi_color[fy:fy+fh, fx:fx+fw]
                
                # Resize for consistency
                if face_roi.size > 0:
                    face_resized = cv2.resize(face_roi, (100, 100))
                    
                    # Match with known faces (with bbox for spatial tracking)
                    person_id = self._match_or_create_person(face_resized, face_roi_color, [x1, y1, x2, y2])
                    
                    recognized_persons.append({
                        'person_id': person_id,
                        'bbox': [x1, y1, x2, y2],
                        'face_bbox': [x1 + fx, y1 + fy, fx + fw, fy + fh],
                        'confidence': 0.9,
                        'detection_method': 'face'
                    })
        
        return recognized_persons
    
    def _match_or_create_person_by_appearance(self, person_roi, person_bbox):
        """
        Match person by overall appearance when face is not detected
        Uses appearance similarity and spatial proximity
        """
        if person_roi.size == 0:
            person_id = f"unknown_{self.person_counter}"
            self.person_counter += 1
            return person_id
        
        # Create appearance histogram (color-based)
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        best_match_id = None
        best_match_score = 0.0
        
        # Compare with known persons
        for person_id, person_data in self.known_faces.items():
            histograms = person_data['histograms']
            last_bbox = person_data.get('last_bbox', None)
            frame_last_seen = person_data.get('frame_last_seen', 0)
            
            # Skip if not seen recently (more than 90 frames ago)
            if self.current_frame_number - frame_last_seen > 90:
                continue
            
            # Kalman filter prediction for better spatial matching
            predicted_bbox = last_bbox
            if person_id in self.kalman_filters:
                kalman = self.kalman_filters[person_id]
                prediction = kalman.predict()
                x, y, w, h = prediction[:4, 0]
                predicted_bbox = [int(x), int(y), int(x+w), int(y+h)]
            
            # Calculate appearance similarity
            max_hist_similarity = 0.0
            for known_hist in histograms:
                similarity = cv2.compareHist(hist, known_hist, cv2.HISTCMP_CORREL)
                max_hist_similarity = max(max_hist_similarity, similarity)
            
            # Calculate spatial proximity bonus (using Kalman prediction)
            spatial_bonus = 0.0
            if predicted_bbox is not None:
                iou = self._calculate_iou(person_bbox, predicted_bbox)
                spatial_bonus = iou * 0.3  # Up to 30% bonus
            
            # Combined score
            total_score = max_hist_similarity + spatial_bonus
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_id = person_id
        
        # Adaptive threshold based on detection history and optimization mode
        match_threshold = self.person_match_threshold  # From optimization mode
        if best_match_id and best_match_id in self.known_faces:
            detection_count = self.known_faces[best_match_id].get('detection_count', 0)
            # Lower threshold for well-established persons (seen 10+ times)
            if detection_count > 10:
                match_threshold *= 0.9  # 10% easier to match
        
        if best_match_id and best_match_score > match_threshold:
            # Update person data with temporal smoothing
            self.known_faces[best_match_id]['histograms'].append(hist)
            self.known_faces[best_match_id]['last_bbox'] = person_bbox
            self.known_faces[best_match_id]['frame_last_seen'] = self.current_frame_number
            self.known_faces[best_match_id]['detection_count'] = self.known_faces[best_match_id].get('detection_count', 0) + 1
            
            # Update Kalman filter
            if best_match_id in self.kalman_filters:
                x1, y1, x2, y2 = person_bbox
                measurement = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
                self.kalman_filters[best_match_id].correct(measurement)
            else:
                self.kalman_filters[best_match_id] = self.create_kalman_filter(person_bbox)
            
            # Track confidence history for temporal smoothing
            if 'confidence_history' not in self.known_faces[best_match_id]:
                self.known_faces[best_match_id]['confidence_history'] = []
            self.known_faces[best_match_id]['confidence_history'].append(best_match_score)
            if len(self.known_faces[best_match_id]['confidence_history']) > 20:
                self.known_faces[best_match_id]['confidence_history'] = self.known_faces[best_match_id]['confidence_history'][-20:]
            
            # Keep only recent histograms (increased to 15 for better matching)
            if len(self.known_faces[best_match_id]['histograms']) > 15:
                self.known_faces[best_match_id]['histograms'] = self.known_faces[best_match_id]['histograms'][-15:]
            
            return best_match_id
        else:
            # New person
            new_id = f"person_{self.person_counter:03d}"
            self.person_counter += 1
            self.known_faces[new_id] = {
                'histograms': [hist],
                'last_bbox': person_bbox,
                'frame_last_seen': self.current_frame_number,
                'detection_count': 1,
                'confidence_history': [best_match_score if best_match_score > 0 else 0.5]
            }
            self.person_logs[new_id] = []
            
            # Initialize Kalman filter for new person
            self.kalman_filters[new_id] = self.create_kalman_filter(person_bbox)
            
            # Save person image
            person_img_path = self.face_db_path / f"{new_id}_appearance.jpg"
            cv2.imwrite(str(person_img_path), person_roi)
            
            # Add to database
            if self.db:
                self.db.add_person(new_id, 'appearance', str(person_img_path))
            
            return new_id
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _match_or_create_person(self, face_roi, person_roi, person_bbox):
        """
        Match face with known persons or create new person ID
        Uses both appearance similarity and spatial proximity
        """
        face_hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
        face_hist = cv2.normalize(face_hist, face_hist).flatten()
        
        best_match_id = None
        best_match_score = 0.0
        
        # Compare with known faces
        for person_id, person_data in self.known_faces.items():
            histograms = person_data['histograms']
            last_bbox = person_data.get('last_bbox', None)
            frame_last_seen = person_data.get('frame_last_seen', 0)
            
            # Skip if not seen recently (more than 90 frames ago at 30fps = 3 seconds)
            if self.current_frame_number - frame_last_seen > 90:
                continue
            
            # Calculate appearance similarity
            max_hist_similarity = 0.0
            for known_hist in histograms:
                similarity = cv2.compareHist(face_hist, known_hist, cv2.HISTCMP_CORREL)
                max_hist_similarity = max(max_hist_similarity, similarity)
            
            # Calculate spatial proximity bonus
            spatial_bonus = 0.0
            if last_bbox is not None:
                iou = self._calculate_iou(person_bbox, last_bbox)
                spatial_bonus = iou * 0.3  # Up to 30% bonus for spatial proximity
            
            # Combined score
            total_score = max_hist_similarity + spatial_bonus
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_id = person_id
        
        # Lowered threshold to 0.6 for better matching
        if best_match_id and best_match_score > 0.6:
            # Match found - update person data
            self.known_faces[best_match_id]['histograms'].append(face_hist)
            self.known_faces[best_match_id]['last_bbox'] = person_bbox
            self.known_faces[best_match_id]['frame_last_seen'] = self.current_frame_number
            
            # Keep only recent histograms (last 10)
            if len(self.known_faces[best_match_id]['histograms']) > 10:
                self.known_faces[best_match_id]['histograms'] = self.known_faces[best_match_id]['histograms'][-10:]
            
            return best_match_id
        else:
            # New person - create ID
            new_id = f"person_{self.person_counter:03d}"
            self.person_counter += 1
            self.known_faces[new_id] = {
                'histograms': [face_hist],
                'last_bbox': person_bbox,
                'frame_last_seen': self.current_frame_number
            }
            self.person_logs[new_id] = []
            
            # Save face image
            face_img_path = self.face_db_path / f"{new_id}_face.jpg"
            cv2.imwrite(str(face_img_path), person_roi)
            
            # Add to database
            if self.db:
                self.db.add_person(new_id, 'face', str(face_img_path))
            
            return new_id
    
    def detect_occupancy(self, results):
        """
        Check if person is present in frame and return person bounding boxes
        """
        person_count = 0
        person_boxes = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            # Class 0 = person in COCO dataset
            if class_id == 0:
                person_count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                person_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        return person_count > 0, person_count, person_boxes
    
    def detect_devices(self, results, frame=None):
        """
        Detect devices with ensemble method for best precision/recall (Enhanced Accuracy)
        """
        devices = []
        device_classes = {
            62: 'laptop',
            63: 'mouse',
            67: 'cell phone',
            72: 'tv'  # Can detect monitors as TVs
        }
        
        for idx, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Apply optimization-mode confidence filtering
            if confidence < self.yolo_conf_threshold:
                continue  # Skip low-confidence detections
            
            if class_id in device_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                device_type = device_classes[class_id]
                
                # Generate unique device ID based on spatial location for temporal tracking
                device_id = f"{device_type}_{int(x1)}_{int(y1)}"
                
                device_info = {
                    'type': device_type,
                    'confidence': confidence,
                    'bbox': bbox,
                    'device_id': device_id
                }
                
                # Detect device state with temporal validation if frame provided
                if frame is not None:
                    state_info = self.energy_analyzer.detect_device_state(frame, bbox, device_type, device_id)
                    device_info.update(state_info)
                    
                    # Ensemble confidence: YOLO + State validation
                    state_confidence = state_info.get('confidence', 0.5)
                    validation_bonus = 0.15 if state_info.get('validated', False) else 0
                    ensemble_confidence = (confidence * 0.6) + (state_confidence * 0.4) + validation_bonus
                    
                    device_info['overall_confidence'] = min(ensemble_confidence, 1.0)
                    
                    # Precision mode: Filter low ensemble confidence
                    if self.optimization_mode == 'precision' and ensemble_confidence < 0.7:
                        continue  # Skip uncertain devices in precision mode
                else:
                    device_info['state'] = 'UNKNOWN'
                    device_info['overall_confidence'] = confidence
                
                devices.append(device_info)
        
        return devices
    
    def log_person_activity(self, person_id, activity_type, details=None):
        """
        Log activity for a specific person (for incentive tracking)
        activity_type: 'entry', 'exit', 'device_usage', 'violation', etc.
        """
        if person_id not in self.person_logs:
            self.person_logs[person_id] = []
        
        activity = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'room_id': self.room_id,
            'details': details
        }
        
        self.person_logs[person_id].append(activity)
        
        # Log to database with incentive points
        if self.db:
            incentive_points = 1 if activity_type == 'presence' else 0
            incentive_reason = 'room_presence' if activity_type == 'presence' else None
            
            self.db.add_activity(
                person_id=person_id,
                activity_type=activity_type,
                details=details,
                incentive_points=incentive_points,
                incentive_reason=incentive_reason,
                room_id=self.room_id
            )
    
    def generate_event(self, occupancy, person_count, devices, recognized_persons=None, video_file=None, frame_number=None, frame=None):
        """
        Create JSON event with enhanced occupancy detection and confidence metrics
        """
        # Enhanced occupancy detection with background subtraction
        if frame is not None:
            # Apply background subtraction for motion-based occupancy verification
            fg_mask = self.bg_subtractor.apply(frame)
            motion_pixels = np.sum(fg_mask == 255)  # Foreground pixels
            total_pixels = fg_mask.size
            motion_ratio = motion_pixels / total_pixels
            
            # Motion-based occupancy (threshold: 1% of frame has motion)
            motion_occupancy = motion_ratio > 0.01
            
            # Ensemble occupancy: Combine YOLO person detection + motion detection
            ensemble_occupancy = occupancy or motion_occupancy
            
            # Temporal smoothing for occupancy (reduce false positives/negatives)
            self.occupancy_buffer.append(ensemble_occupancy)
            if len(self.occupancy_buffer) > self.occupancy_buffer_size:
                self.occupancy_buffer = self.occupancy_buffer[-self.occupancy_buffer_size:]
            
            # Use majority vote from buffer
            if len(self.occupancy_buffer) >= 3:
                occupancy_votes = sum(self.occupancy_buffer)
                smoothed_occupancy = occupancy_votes > (len(self.occupancy_buffer) / 2)
            else:
                smoothed_occupancy = ensemble_occupancy
            
            # Override with smoothed value
            occupancy = smoothed_occupancy
        
        # Separate devices by state
        devices_on = [d for d in devices if d.get('state') == 'ON']
        devices_off = [d for d in devices if d.get('state') == 'OFF']
        
        # Detect lights state if frame provided
        lights_on = False
        if frame is not None:
            lights_info = self.energy_analyzer.detect_lights_state(frame)
            lights_on = lights_info['lights_on']
        
        # Detect sustainable/unsustainable actions
        action_result = self.energy_analyzer.detect_sustainable_action(
            devices, self.previous_devices_state,
            occupancy, self.previous_occupancy
        )
        
        # Calculate energy savings
        energy_metrics = self.energy_analyzer.calculate_energy_savings(
            devices_on, devices_off, duration_minutes=5
        )
        
        # Get location verification status
        location_status = {'verified': self.location_verified, 'confidence': self.location_confidence}
        
        # Calculate overall detection confidence
        person_confidences = [p.get('confidence', 0.5) for p in (recognized_persons or [])]
        avg_person_confidence = sum(person_confidences) / len(person_confidences) if person_confidences else 0.0
        
        device_confidences = [d.get('overall_confidence', d.get('confidence', 0.5)) for d in devices]
        avg_device_confidence = sum(device_confidences) / len(device_confidences) if device_confidences else 0.0
        
        # Overall event confidence (weighted average)
        overall_confidence = (
            location_status['confidence'] * 0.2 +  # 20% location
            avg_person_confidence * 0.4 +          # 40% person recognition
            avg_device_confidence * 0.4            # 40% device detection
        )
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "room_id": self.room_id,
            "department": self.department,  # Campus: track by department
            "location_verified": location_status['verified'],
            "location_confidence": round(location_status['confidence'], 2),
            "overall_confidence": round(overall_confidence, 2),  # NEW: Overall detection confidence
            "occupancy": occupancy,
            "person_count": person_count,
            "recognized_persons": recognized_persons if recognized_persons else [],
            "devices_detected": devices,
            "devices_on": devices_on,
            "devices_off": devices_off,
            "lights_on": lights_on,
            "action_detected": action_result['action_detected'],
            "action_type": action_result['action_type'],
            "action_confidence": action_result.get('confidence', 0.7),  # NEW: Action detection confidence
            "energy_saved_estimate": action_result.get('energy_impact', energy_metrics['power_saved_watts']),
            "blockchain_credits": action_result.get('blockchain_credits', 0.0),
            "waste_multiplier": action_result.get('waste_multiplier', 1.0),  # Campus priority multiplier
            "priority_devices": action_result.get('priority_devices', {}),  # Projector/AC flags
            "video_file": video_file,
            "frame_number": frame_number,
            "energy_metrics": energy_metrics
        }
        
        # Update previous state
        self.previous_devices_state = devices.copy()
        self.previous_occupancy = occupancy
        self.previous_lights_on = lights_on
        
        # Log entry/presence for each recognized person
        if recognized_persons:
            for person in recognized_persons:
                self.log_person_activity(
                    person['person_id'],
                    'presence',
                    {'devices_nearby': len(devices), 'action': action_result['action_detected']}
                )
                
                # Log event to database
                if self.db:
                    event_data = {
                        'timestamp': event['timestamp'],
                        'room_id': self.room_id,
                        'department': self.department,  # Campus department tracking
                        'occupancy': occupancy,
                        'person_count': 1,
                        'person_id': person['person_id'],
                        'bbox': person.get('bbox'),
                        'face_bbox': person.get('face_bbox'),
                        'confidence': person.get('confidence'),
                        'detection_method': person.get('detection_method'),
                        'devices_detected': devices,
                        'devices_on': devices_on,
                        'devices_off': devices_off,
                        'lights_on': lights_on,
                        'video_file': video_file,
                        'frame_number': frame_number,
                        'action_detected': action_result['action_detected'],
                        'action_type': action_result['action_type'],
                        'energy_saved_estimate': event['energy_saved_estimate'],
                        'blockchain_credits': event['blockchain_credits']
                    }
                    self.db.add_event(event_data)
        
        return event
    
    def save_person_logs(self, output_path='outputs/person_logs.json'):
        """
        Save all person activity logs to JSON file (Campus optimized)
        """
        logs_summary = {
            'total_persons_tracked': len(self.person_logs),
            'room_id': self.room_id,
            'department': self.department,  # Campus department
            'generated_at': datetime.now().isoformat(),
            'person_logs': {}
        }
        
        for person_id, activities in self.person_logs.items():
            logs_summary['person_logs'][person_id] = {
                'total_activities': len(activities),
                'first_seen': activities[0]['timestamp'] if activities else None,
                'last_seen': activities[-1]['timestamp'] if activities else None,
                'activities': activities
            }
        
        with open(output_path, 'w') as f:
            json.dump(logs_summary, f, indent=2)
        
        print(f"✓ Saved person logs to {output_path}")
        return logs_summary
    
    def process_video(self, video_source, mode='realtime'):
        """
        Process video stream (realtime=0 for webcam, or file path)
        mode: 'realtime' or 'uploaded'
        """
        # Get video filename
        video_filename = Path(video_source).name if mode == 'uploaded' else 'webcam'
        
        # Open video source
        if mode == 'realtime':
            cap = cv2.VideoCapture(0)  # Webcam
        else:
            cap = cv2.VideoCapture(video_source)  # Video file
        
        if not cap.isOpened():
            print("Error: Cannot open video source")
            return
        
        events = []
        frame_count = 0
        self.current_frame_number = 0  # Reset frame counter for tracking
        
        print(f"Processing video ({mode})... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.current_frame_number = frame_count
            
            # Process every 30th frame to save computation (5-second intervals at 30fps)
            # For 5-second intervals: 30fps * 5s = 150 frames
            processing_interval = 150  # Process every 5 seconds
            
            if frame_count % processing_interval == 0:
                # Enhance frame for low-light conditions
                enhanced_frame = self.energy_analyzer.enhance_low_light_frame(frame)
                
                results = self.process_frame(enhanced_frame)
                occupancy, person_count, person_boxes = self.detect_occupancy(results)
                devices = self.detect_devices(results, frame=enhanced_frame)  # Pass frame for state detection
                
                # Recognize persons with face detection
                recognized_persons = []
                if person_count > 0:
                    recognized_persons = self.detect_and_recognize_faces(enhanced_frame, person_boxes)
                
                # Generate event with person recognition and energy tracking (includes DB logging)
                event = self.generate_event(occupancy, person_count, devices, recognized_persons, video_filename, frame_count, frame=enhanced_frame)
                events.append(event)
                
                # Print comprehensive status
                person_ids = [p['person_id'] for p in recognized_persons]
                devices_on = len([d for d in devices if d.get('state') == 'ON'])
                devices_total = len(devices)
                action = event.get('action_type', 'neutral')
                credits = event.get('blockchain_credits', 0)
                
                print(f"Frame {frame_count}: {person_count} person(s) {person_ids}, "
                      f"{devices_on}/{devices_total} devices ON, "
                      f"Action: {action}, Credits: ₹{credits}")
                
                # Draw detections on frame
                annotated_frame = results.plot()
                
                # Draw person IDs on frame
                for person in recognized_persons:
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, person['person_id'], (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw device states
                for device in devices:
                    if 'bbox' in device:
                        dx1, dy1, dx2, dy2 = device['bbox']
                        state = device.get('state', 'UNKNOWN')
                        color = (0, 255, 0) if state == 'ON' else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (dx1, dy1), (dx2, dy2), color, 1)
                        cv2.putText(annotated_frame, f"{device['type']}: {state}", 
                                   (dx1, dy1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                cv2.imshow('SCA - CV Detection', annotated_frame)
            
            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save events to JSON
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"outputs/events_{timestamp_str}.json"
        with open(output_file, 'w') as f:
            json.dump(events, f, indent=2)
        
        # Save person logs
        person_logs_file = f"outputs/person_logs_{timestamp_str}.json"
        self.save_person_logs(person_logs_file)
        
        print(f"✓ Saved {len(events)} events to {output_file}")
        print(f"✓ Tracked {len(self.known_faces)} unique person(s)")
        return events

# Test the processor
if __name__ == "__main__":
    from pathlib import Path
    
    processor = CVProcessor()
    
    # Find all test videos in current directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    current_dir = Path('.')
    test_videos = []
    
    for ext in video_extensions:
        test_videos.extend(current_dir.glob(f'*{ext}'))
    
    if not test_videos:
        print("No test videos found in current directory")
        print("Please add video files (.mp4, .avi, .mov, .mkv, .webm)")
    else:
        print(f"Found {len(test_videos)} video(s) to process\n")
        
        for i, video_path in enumerate(test_videos, 1):
            print(f"\n{'='*60}")
            print(f"Processing Video {i}/{len(test_videos)}: {video_path.name}")
            print(f"{'='*60}")
            
            try:
                events = processor.process_video(str(video_path), mode='uploaded')
                print(f"✓ Successfully processed {video_path.name}")
                print(f"  - Generated {len(events)} events")
            except Exception as e:
                print(f"✗ Failed to process {video_path.name}: {e}")
        
        print(f"\n{'='*60}")
        print("Batch processing complete!")
        print("Check the 'outputs/' folder for JSON event files")
        print(f"{'='*60}")
