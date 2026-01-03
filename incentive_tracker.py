"""
Incentive Tracker for Person Activity Logging
Tracks positive and negative incentives for each recognized person
"""
import json
from datetime import datetime
from pathlib import Path

class IncentiveTracker:
    def __init__(self, person_logs_file=None):
        """
        Initialize incentive tracker
        
        Args:
            person_logs_file: Path to person logs JSON file
        """
        self.incentives = {}  # {person_id: {'positive': [], 'negative': []}}
        
        if person_logs_file:
            self.load_person_logs(person_logs_file)
    
    def load_person_logs(self, logs_file):
        """Load person activity logs from JSON file"""
        with open(logs_file, 'r') as f:
            data = json.load(f)
        
        self.person_logs = data.get('person_logs', {})
        
        # Initialize incentives for each person
        for person_id in self.person_logs.keys():
            if person_id not in self.incentives:
                self.incentives[person_id] = {'positive': [], 'negative': []}
    
    def add_positive_incentive(self, person_id, reason, points=1):
        """
        Add positive incentive for a person
        
        Args:
            person_id: Person identifier
            reason: Reason for incentive (e.g., "present_on_time", "device_usage")
            points: Points awarded
        """
        if person_id not in self.incentives:
            self.incentives[person_id] = {'positive': [], 'negative': []}
        
        incentive = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'points': points
        }
        
        self.incentives[person_id]['positive'].append(incentive)
        print(f"✓ +{points} points for {person_id}: {reason}")
    
    def add_negative_incentive(self, person_id, reason, points=1):
        """
        Add negative incentive for a person
        
        Args:
            person_id: Person identifier
            reason: Reason for penalty (e.g., "absent", "violation")
            points: Points deducted
        """
        if person_id not in self.incentives:
            self.incentives[person_id] = {'positive': [], 'negative': []}
        
        incentive = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'points': points
        }
        
        self.incentives[person_id]['negative'].append(incentive)
        print(f"✗ -{points} points for {person_id}: {reason}")
    
    def get_person_score(self, person_id):
        """Calculate total score for a person"""
        if person_id not in self.incentives:
            return 0
        
        positive_total = sum(i['points'] for i in self.incentives[person_id]['positive'])
        negative_total = sum(i['points'] for i in self.incentives[person_id]['negative'])
        
        return positive_total - negative_total
    
    def get_leaderboard(self):
        """Get leaderboard sorted by score"""
        leaderboard = []
        
        for person_id in self.incentives.keys():
            score = self.get_person_score(person_id)
            positive = len(self.incentives[person_id]['positive'])
            negative = len(self.incentives[person_id]['negative'])
            
            leaderboard.append({
                'person_id': person_id,
                'score': score,
                'positive_count': positive,
                'negative_count': negative
            })
        
        # Sort by score (descending)
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        return leaderboard
    
    def analyze_activity_patterns(self, person_id):
        """
        Analyze activity patterns and auto-assign incentives
        
        Rules:
        - Present in room = +1 point per detection
        - Device usage detected = +1 point
        - Long absence after entry = -2 points
        """
        if person_id not in self.person_logs:
            print(f"No logs found for {person_id}")
            return
        
        activities = self.person_logs[person_id]['activities']
        
        for activity in activities:
            activity_type = activity['activity_type']
            
            if activity_type == 'presence':
                # Positive: person is present
                self.add_positive_incentive(person_id, 'room_presence', 1)
                
                # Check device usage
                details = activity.get('details', {})
                devices_nearby = details.get('devices_nearby', 0)
                if devices_nearby > 0:
                    self.add_positive_incentive(person_id, 'device_detected', 1)
    
    def save_incentives(self, output_path='outputs/incentives.json'):
        """Save incentive data to JSON"""
        leaderboard = self.get_leaderboard()
        
        output = {
            'generated_at': datetime.now().isoformat(),
            'total_persons': len(self.incentives),
            'leaderboard': leaderboard,
            'detailed_incentives': self.incentives
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved incentives to {output_path}")
        return output
    
    def print_summary(self):
        """Print incentive summary"""
        print("\n" + "=" * 60)
        print("INCENTIVE LEADERBOARD")
        print("=" * 60)
        
        leaderboard = self.get_leaderboard()
        
        if not leaderboard:
            print("No data available")
            return
        
        header = f"{'Rank':<6} {'Person ID':<15} {'Score':<8} {'(+/-)'}"
        print(header)
        print("-" * 60)
        
        for i, entry in enumerate(leaderboard, 1):
            person_id = entry['person_id']
            score = entry['score']
            pos = entry['positive_count']
            neg = entry['negative_count']
            
            print(f"{i:<6} {person_id:<15} {score:<8} (+{pos}/-{neg})")
        
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Find latest person logs file
    outputs_dir = Path('outputs')
    log_files = sorted(outputs_dir.glob('person_logs_*.json'), reverse=True)
    
    if not log_files:
        print("No person logs found. Run cv_processor.py first.")
        sys.exit(1)
    
    latest_log = log_files[0]
    print(f"Loading logs from: {latest_log}")
    
    # Initialize tracker
    tracker = IncentiveTracker(latest_log)
    
    # Analyze all persons
    print("\nAnalyzing activity patterns...")
    for person_id in tracker.person_logs.keys():
        tracker.analyze_activity_patterns(person_id)
    
    # Print summary
    tracker.print_summary()
    
    # Save results
    tracker.save_incentives('outputs/incentives.json')
