"""
Energy Analyzer Module for Sustainable Campus Automation
Detects device states, calculates energy savings, and assigns blockchain credits
"""
import cv2
import numpy as np
from datetime import datetime
import json

class EnergyAnalyzer:
    """Analyzes energy consumption and sustainability actions"""
    
    # Energy consumption values (Watts) - Campus optimized
    DEVICE_POWER = {
        'laptop': 45,           # Student laptop
        'desktop': 100,         # Lab desktop computer
        'monitor': 30,          # LCD monitor
        'tv': 80,              # TV/large monitor
        'projector': 250,       # Classroom projector (HIGH PRIORITY)
        'cell phone': 5,        # Phone charger
        'mouse': 1,            # Wireless mouse
        'light_bulb': 20,      # LED bulb
        'tube_light': 40,      # Tube light (4x in classroom = 160W)
        'fan': 75,             # Ceiling fan (4x in classroom = 300W)
        'ac': 1500,            # Air conditioner (HIGHEST PRIORITY)
        'printer': 50,         # Laser printer in lab
        'router': 15,          # Network router
        'server': 300          # Lab server
    }
    
    # Blockchain credit rates (₹ per kWh saved)
    CREDIT_RATE_PER_KWH = 5.0  # ₹5 per kWh
    
    # Campus timing patterns
    CLASS_HOURS_START = 8    # 8 AM
    CLASS_HOURS_END = 18     # 6 PM
    BREAK_DURATION = 0.25    # 15 minutes between classes
    LAB_SESSION_HOURS = 3    # Typical lab session duration
    
    # Detection thresholds
    LIGHT_THRESHOLD_DAY = 120      # Brightness threshold for daytime
    LIGHT_THRESHOLD_NIGHT = 40     # Brightness threshold for nighttime
    SCREEN_ON_THRESHOLD = 30       # Minimum brightness for screen ON
    
    # Campus-specific waste thresholds
    PROJECTOR_WASTE_PRIORITY = 2.0   # 2x credits for projector waste
    AC_WASTE_PRIORITY = 3.0          # 3x credits for AC waste
    EMPTY_CLASSROOM_MULTIPLIER = 1.5  # 1.5x for empty classrooms
    
    def __init__(self, room_id="CS_LAB_101", optimization_mode='balanced'):
        self.room_id = room_id
        self.previous_state = {}  # Track previous device states
        self.action_history = []  # Track actions for pattern analysis
        self.department = room_id.split('_')[0] if '_' in room_id else 'GENERAL'  # Extract dept from room_id
        
        # Optimization mode for precision/recall trade-off
        self.optimization_mode = optimization_mode
        
        # Temporal smoothing for action detection (reduce false positives)
        self.device_state_buffer = {}  # Buffer device states: {device_id: [recent_states]}
        self.occupancy_buffer = []  # Buffer occupancy states for smoothing
        
        # Adaptive buffer size based on optimization mode
        if optimization_mode == 'precision':
            self.buffer_size = 5  # More frames for high precision
            self.action_confidence_threshold = 0.80
        elif optimization_mode == 'recall':
            self.buffer_size = 2  # Fewer frames for high recall
            self.action_confidence_threshold = 0.60
        else:  # balanced
            self.buffer_size = 3  # Balanced
            self.action_confidence_threshold = 0.70
        
    def detect_device_state(self, frame, device_bbox, device_type, device_id=None):
        """
        Detect if a device is ON or OFF with multi-frame validation for accuracy
        Optimized with vectorized operations and reduced allocations
        
        Args:
            frame: Video frame
            device_bbox: Bounding box [x1, y1, x2, y2]
            device_type: Type of device
            device_id: Unique device identifier for temporal tracking
            
        Returns:
            dict: {state: 'ON'/'OFF', confidence: float, brightness: int, validated: bool}
        """
        x1, y1, x2, y2 = device_bbox
        
        # Extract device region (avoid copy when possible)
        device_roi = frame[y1:y2, x1:x2]
        
        if device_roi.size == 0:
            return {'state': 'UNKNOWN', 'confidence': 0.0, 'brightness': 0}
        
        # Vectorized brightness calculation (faster than grayscale conversion)
        # Use mean across color channels directly
        avg_brightness = device_roi.mean()
        max_brightness = device_roi.max()
        
        # Detect screen-like devices (laptop, monitor, tv, projector)
        if device_type in ['laptop', 'monitor', 'tv', 'desktop', 'projector']:
            # Check for bright regions (screen on)
            bright_pixels = np.sum(gray > self.SCREEN_ON_THRESHOLD)
            total_pixels = gray.size
            bright_ratio = bright_pixels / total_pixels
            
            # Projectors have higher brightness detection
            threshold = 0.3 if device_type != 'projector' else 0.2
            brightness_threshold = 150 if device_type != 'projector' else 180
            
            if bright_ratio > threshold or max_brightness > brightness_threshold:
                # Screen is ON
                confidence = min(bright_ratio * 2, 1.0)
                detected_state = 'ON'
            else:
                detected_state = 'OFF'
                confidence = 0.7
            
            # Multi-frame validation for higher accuracy
            validated = False
            if device_id:
                if device_id not in self.device_state_buffer:
                    self.device_state_buffer[device_id] = []
                
                # Add current detection to buffer
                self.device_state_buffer[device_id].append(detected_state)
                if len(self.device_state_buffer[device_id]) > self.buffer_size:
                    self.device_state_buffer[device_id] = self.device_state_buffer[device_id][-self.buffer_size:]
                
                # Validate if majority of recent detections agree
                if len(self.device_state_buffer[device_id]) >= self.buffer_size:
                    state_counts = {}
                    for state in self.device_state_buffer[device_id]:
                        state_counts[state] = state_counts.get(state, 0) + 1
                    most_common_state = max(state_counts, key=state_counts.get)
                    
                    # Adaptive consensus requirement based on optimization mode
                    required_consensus = self.buffer_size - 1 if self.optimization_mode == 'precision' else max(self.buffer_size // 2, 1)
                    
                    if state_counts[most_common_state] >= required_consensus:
                        validated = True
                        detected_state = most_common_state
                        confidence = min(confidence + 0.25, 1.0)  # Boost confidence for validated states
            
            if detected_state == 'ON':
                return {
                    'state': 'ON',
                    'confidence': confidence,
                    'brightness': int(avg_brightness),
                    'priority': 'HIGH' if device_type == 'projector' else 'NORMAL',
                    'method': 'screen_brightness',
                    'validated': validated
                }
            else:
                return {
                    'state': 'OFF',
                    'confidence': confidence,
                    'brightness': int(avg_brightness),
                    'method': 'screen_darkness',
                    'validated': validated
                }
        
        # Other devices - use general brightness
        elif device_type == 'cell phone':
            if avg_brightness > 80:
                return {'state': 'CHARGING', 'confidence': 0.6, 'brightness': int(avg_brightness)}
            else:
                return {'state': 'OFF', 'confidence': 0.5, 'brightness': int(avg_brightness)}
        
        return {'state': 'UNKNOWN', 'confidence': 0.3, 'brightness': int(avg_brightness)}
    
    def detect_lights_state(self, frame):
        """
        Detect if room lights are ON or OFF
        
        Args:
            frame: Video frame
            
        Returns:
            dict: {lights_on: bool, confidence: float, ambient_brightness: int}
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Check time of day (simplified - would use actual time in production)
        current_hour = datetime.now().hour
        is_daytime = 6 <= current_hour <= 18
        
        # Determine threshold based on time
        threshold = self.LIGHT_THRESHOLD_DAY if is_daytime else self.LIGHT_THRESHOLD_NIGHT
        
        # Detect lights state
        if avg_brightness > threshold:
            confidence = min((avg_brightness - threshold) / threshold, 1.0)
            return {
                'lights_on': True,
                'confidence': confidence,
                'ambient_brightness': int(avg_brightness),
                'time_context': 'day' if is_daytime else 'night'
            }
        else:
            confidence = min((threshold - avg_brightness) / threshold, 1.0)
            return {
                'lights_on': False,
                'confidence': confidence,
                'ambient_brightness': int(avg_brightness),
                'time_context': 'day' if is_daytime else 'night'
            }
    
    def detect_sustainable_action(self, current_devices, previous_devices, occupancy, previous_occupancy):
        """
        Detect sustainable or unsustainable actions with confidence scoring
        
        Args:
            current_devices: Current detected devices
            previous_devices: Previously detected devices
            occupancy: Current occupancy status
            previous_occupancy: Previous occupancy status
            
        Returns:
            dict: {action_type, action_detected, energy_impact, credits, confidence}
        """
        action = {
            'action_type': 'neutral',
            'action_detected': None,
            'energy_impact': 0.0,
            'blockchain_credits': 0.0,
            'confidence': 0.5,  # Base confidence
            'description': ''
        }
        
        # Calculate device state validation confidence
        validated_devices = [d for d in current_devices if d.get('validated', False)]
        device_confidence = len(validated_devices) / len(current_devices) if current_devices else 0.5
        
        # Scenario 1: Person left room but devices still ON (UNSUSTAINABLE - Campus Priority)
        if not occupancy and previous_occupancy:
            devices_left_on = self._count_devices_on(current_devices)
            if devices_left_on > 0:
                energy_wasted = self._calculate_device_power(current_devices)
                
                # Campus-specific: Check for high-priority devices
                has_projector = any(d.get('type') == 'projector' and d.get('state') == 'ON' for d in current_devices)
                has_ac = any(d.get('type') == 'ac' and d.get('state') == 'ON' for d in current_devices)
                
                # Apply priority multipliers
                multiplier = self.EMPTY_CLASSROOM_MULTIPLIER
                if has_projector:
                    multiplier *= self.PROJECTOR_WASTE_PRIORITY
                if has_ac:
                    multiplier *= self.AC_WASTE_PRIORITY
                
                # Calculate confidence based on device validation
                action_confidence = 0.6 + (device_confidence * 0.4)  # 60-100% confidence
                
                action['action_type'] = 'unsustainable'
                action['action_detected'] = 'empty_classroom_waste' if (has_projector or has_ac) else 'devices_left_on_empty_room'
                action['energy_impact'] = -energy_wasted
                action['waste_multiplier'] = multiplier
                action['confidence'] = round(action_confidence, 2)
                action['priority_devices'] = {
                    'projector_on': has_projector,
                    'ac_on': has_ac
                }
                action['description'] = f'{devices_left_on} device(s) left ON in empty classroom (Priority: {"HIGH" if multiplier > 2 else "MEDIUM"})'
                return action
        
        # Scenario 2: Devices turned OFF when leaving (SUSTAINABLE - Campus Bonus)
        prev_on = self._count_devices_on(previous_devices)
        curr_on = self._count_devices_on(current_devices)
        
        if prev_on > curr_on and not occupancy:
            devices_turned_off = prev_on - curr_on
            energy_saved = self._calculate_power_difference(previous_devices, current_devices)
            
            # Campus bonus for turning off high-priority devices
            has_projector = any(d.get('type') == 'projector' and d.get('state') == 'OFF' for d in current_devices)
            has_ac = any(d.get('type') == 'ac' and d.get('state') == 'OFF' for d in current_devices)
            multiplier = 1.5 if (has_projector or has_ac) else 1.0
            
            # Higher confidence for well-validated state changes
            action_confidence = 0.7 + (device_confidence * 0.3)
            
            action['action_type'] = 'sustainable'
            action['action_detected'] = 'devices_turned_off_on_exit'
            action['energy_impact'] = energy_saved
            action['confidence'] = round(action_confidence, 2)
            action['blockchain_credits'] = self._calculate_credits(energy_saved, duration_hours=self.LAB_SESSION_HOURS, multiplier=multiplier)
            action['description'] = f'{devices_turned_off} device(s) turned OFF on exit (Campus bonus: {multiplier}x)'
            return action
        
        # Scenario 3: Devices turned ON unnecessarily (UNSUSTAINABLE)
        if curr_on > prev_on and not occupancy:
            action_confidence = 0.65 + (device_confidence * 0.35)
            action['action_type'] = 'unsustainable'
            action['action_detected'] = 'devices_turned_on_empty_room'
            action['energy_impact'] = -self._calculate_power_difference(current_devices, previous_devices)
            action['confidence'] = round(action_confidence, 2)
            action['description'] = 'Devices turned ON in empty room'
            return action
        
        # Scenario 4: Efficient device usage during class (SUSTAINABLE)
        if occupancy and curr_on < prev_on:
            energy_saved = self._calculate_power_difference(previous_devices, current_devices)
            action_confidence = 0.75 + (device_confidence * 0.25)  # Higher base confidence for this scenario
            action['action_type'] = 'sustainable'
            action['action_detected'] = 'efficient_device_usage'
            action['energy_impact'] = energy_saved
            action['confidence'] = round(action_confidence, 2)
            action['blockchain_credits'] = self._calculate_credits(energy_saved, duration_hours=self.LAB_SESSION_HOURS)
            action['description'] = 'Device(s) turned OFF during class (efficient usage)'
            return action
        
        return action
    
    def _count_devices_on(self, devices):
        """Count devices in ON state"""
        count = 0
        for device in devices:
            if device.get('state') == 'ON':
                count += 1
        return count
    
    def _calculate_device_power(self, devices):
        """Calculate total power consumption of devices"""
        total_power = 0.0
        for device in devices:
            if device.get('state') == 'ON':
                device_type = device.get('type', 'laptop')
                power = self.DEVICE_POWER.get(device_type, 30)
                total_power += power
        return total_power
    
    def _calculate_power_difference(self, devices_before, devices_after):
        """Calculate power difference between two device states"""
        power_before = self._calculate_device_power(devices_before)
        power_after = self._calculate_device_power(devices_after)
        return power_before - power_after
    
    def _calculate_credits(self, watts_saved, duration_hours=1.0, multiplier=1.0):
        """
        Calculate blockchain credits based on energy saved (Campus optimized)
        
        Args:
            watts_saved: Power saved in Watts
            duration_hours: Duration in hours (campus default: 3 hours for lab session)
            multiplier: Priority multiplier for high-waste scenarios
            
        Returns:
            float: Credits in ₹
        """
        if watts_saved <= 0:
            return 0.0
        
        # Campus timing: Use lab session duration if not specified
        if duration_hours == 1.0:
            current_hour = datetime.now().hour
            if self.CLASS_HOURS_START <= current_hour <= self.CLASS_HOURS_END:
                duration_hours = self.LAB_SESSION_HOURS  # 3-hour lab session
            else:
                duration_hours = 0.5  # After-hours usage
        
        # Convert to kWh
        kwh_saved = (watts_saved * duration_hours) / 1000
        
        # Calculate credits with campus multiplier
        credits = kwh_saved * self.CREDIT_RATE_PER_KWH * multiplier
        
        # Round to 2 decimal places
        return round(credits, 2)
    
    def calculate_energy_savings(self, devices_on, devices_off, duration_minutes=5):
        """
        Calculate energy savings and blockchain credits
        
        Args:
            devices_on: List of devices in ON state
            devices_off: List of devices in OFF state
            duration_minutes: Time period in minutes
            
        Returns:
            dict: Energy metrics and credits
        """
        # Calculate power for ON devices
        power_consumed = sum([self.DEVICE_POWER.get(d.get('type', 'laptop'), 30) 
                             for d in devices_on])
        
        # Calculate potential savings from OFF devices
        power_saved = sum([self.DEVICE_POWER.get(d.get('type', 'laptop'), 30) 
                          for d in devices_off])
        
        # Convert to kWh
        duration_hours = duration_minutes / 60
        energy_consumed_kwh = (power_consumed * duration_hours) / 1000
        energy_saved_kwh = (power_saved * duration_hours) / 1000
        
        # Calculate credits for saved energy
        credits = self._calculate_credits(power_saved, duration_hours)
        
        return {
            'power_consumed_watts': round(power_consumed, 2),
            'power_saved_watts': round(power_saved, 2),
            'energy_consumed_kwh': round(energy_consumed_kwh, 4),
            'energy_saved_kwh': round(energy_saved_kwh, 4),
            'blockchain_credits': credits,
            'duration_minutes': duration_minutes,
            'potential_monthly_savings_rupees': round(credits * 12 * 30, 2)  # Extrapolated
        }
    
    def enhance_low_light_frame(self, frame):
        """
        Enhance frame for better detection in low-light conditions
        
        Args:
            frame: Input video frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        L_enhanced = clahe.apply(L_channel)
        
        # Merge channels
        enhanced_lab = cv2.merge([L_enhanced, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def generate_energy_report(self, events, time_period_hours=24):
        """
        Generate energy usage report from events
        
        Args:
            events: List of event dictionaries
            time_period_hours: Time period for report
            
        Returns:
            dict: Comprehensive energy report
        """
        total_credits = sum([e.get('blockchain_credits', 0) for e in events])
        total_energy_saved = sum([e.get('energy_saved_estimate', 0) for e in events])
        
        sustainable_actions = [e for e in events if e.get('action_type') == 'sustainable']
        unsustainable_actions = [e for e in events if e.get('action_type') == 'unsustainable']
        
        report = {
            'time_period_hours': time_period_hours,
            'total_events': len(events),
            'total_blockchain_credits': round(total_credits, 2),
            'total_energy_saved_watts': round(total_energy_saved, 2),
            'sustainable_actions_count': len(sustainable_actions),
            'unsustainable_actions_count': len(unsustainable_actions),
            'sustainability_score': self._calculate_sustainability_score(events),
            'estimated_cost_savings_rupees': round(total_credits, 2),
            'projected_annual_savings': round(total_credits * (8760 / time_period_hours), 2)
        }
        
        return report
    
    def generate_performance_metrics(self):
        """
        Generate comprehensive performance metrics report
        Returns: dict with precision, recall, F1, specificity, accuracy
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1_score()
        specificity = self.calculate_specificity()
        
        # Calculate accuracy if we have ground truth data
        tp = sum([e.get('true_positive', 0) for e in self.action_history])
        tn = sum([e.get('true_negative', 0) for e in self.action_history])
        fp = sum([e.get('false_positive', 0) for e in self.action_history])
        fn = sum([e.get('false_negative', 0) for e in self.action_history])
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        return {
            'optimization_mode': self.optimization_mode,
            'metrics': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'specificity': round(specificity, 4)
            },
            'counts': {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            },
            'sample_size': len(self.action_history),
            'buffer_size': self.buffer_size,
            'confidence_threshold': self.action_confidence_threshold
        }
    
    def _calculate_sustainability_score(self, events):
        """
        Calculate weighted sustainability score (0-100) with priority device bonuses
        Higher accuracy through weighted scoring based on impact
        """
        if not events:
            return 0
        
        total_positive_weight = 0.0
        total_negative_weight = 0.0
        
        for event in events:
            action_type = event.get('action_type')
            energy_impact = abs(event.get('energy_saved_estimate', 0))
            
            # Weight by energy impact (higher impact = more weight)
            weight = 1.0 + (energy_impact / 1000.0)  # Base weight + energy factor
            
            # Additional weight for priority devices
            priority_devices = event.get('priority_devices', {})
            if priority_devices.get('projector_on') or priority_devices.get('ac_on'):
                weight *= 1.5  # 1.5x weight for high-priority devices
            
            if action_type == 'sustainable':
                total_positive_weight += weight
            elif action_type == 'unsustainable':
                total_negative_weight += weight
        
        if total_positive_weight + total_negative_weight == 0:
            return 50  # Neutral
        
        # Calculate weighted score
        score = (total_positive_weight / (total_positive_weight + total_negative_weight)) * 100
        
        # Apply confidence adjustment based on sample size
        confidence_factor = min(len(events) / 20.0, 1.0)  # Full confidence at 20+ events
        adjusted_score = 50 + (score - 50) * confidence_factor  # Regress toward neutral for low samples
        
        return round(adjusted_score, 1)
    
    def calculate_precision(self):
        """Calculate detection precision: TP / (TP + FP)"""
        tp = sum([e.get('true_positive', 0) for e in self.action_history])
        fp = sum([e.get('false_positive', 0) for e in self.action_history])
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def calculate_recall(self):
        """Calculate detection recall: TP / (TP + FN)"""
        tp = sum([e.get('true_positive', 0) for e in self.action_history])
        fn = sum([e.get('false_negative', 0) for e in self.action_history])
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def calculate_f1_score(self):
        """Calculate F1 score: 2 × (Precision × Recall) / (Precision + Recall)"""
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def calculate_specificity(self):
        """Calculate specificity: TN / (TN + FP)"""
        tn = sum([e.get('true_negative', 0) for e in self.action_history])
        fp = sum([e.get('false_positive', 0) for e in self.action_history])
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0


if __name__ == "__main__":
    # Test energy analyzer with campus scenario
    analyzer = EnergyAnalyzer("CS_LAB_101")
    
    print("=" * 60)
    print("CAMPUS ENERGY ANALYZER - TEST")
    print(f"Department: {analyzer.department}")
    print(f"Room: {analyzer.room_id}")
    print("=" * 60)
    print("ENERGY ANALYZER TEST")
    print("=" * 60)
    
    # Test energy calculation
    devices_on = [
        {'type': 'laptop', 'state': 'ON'},
        {'type': 'monitor', 'state': 'ON'}
    ]
    
    devices_off = [
        {'type': 'laptop', 'state': 'OFF'},
        {'type': 'tv', 'state': 'OFF'}
    ]
    
    metrics = analyzer.calculate_energy_savings(devices_on, devices_off, duration_minutes=5)
    
    print("\nEnergy Metrics (5-minute interval):")
    print(f"  Power Consumed: {metrics['power_consumed_watts']} W")
    print(f"  Power Saved: {metrics['power_saved_watts']} W")
    print(f"  Energy Saved: {metrics['energy_saved_kwh']} kWh")
    print(f"  Blockchain Credits: ₹{metrics['blockchain_credits']}")
    print(f"  Projected Monthly Savings: ₹{metrics['potential_monthly_savings_rupees']}")
    
    # Test action detection
    previous_devices = [
        {'type': 'laptop', 'state': 'ON'},
        {'type': 'monitor', 'state': 'ON'}
    ]
    
    current_devices = [
        {'type': 'laptop', 'state': 'OFF'},
        {'type': 'monitor', 'state': 'OFF'}
    ]
    
    action = analyzer.detect_sustainable_action(
        current_devices, previous_devices,
        occupancy=False, previous_occupancy=True
    )
    
    print("\nSustainable Action Detection:")
    print(f"  Action Type: {action['action_type']}")
    print(f"  Action: {action['action_detected']}")
    print(f"  Energy Impact: {action['energy_impact']} W")
    print(f"  Credits Earned: ₹{action['blockchain_credits']}")
    print(f"  Description: {action['description']}")
    
    print("\n" + "=" * 60)
