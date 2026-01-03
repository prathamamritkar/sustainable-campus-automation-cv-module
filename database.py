"""
Database models and configuration for SCA CV Module
Uses SQLAlchemy with SQLite
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from datetime import datetime
import json

Base = declarative_base()

class Person(Base):
    """Person tracking table - Campus optimized"""
    __tablename__ = 'persons'
    
    person_id = Column(String(50), primary_key=True)  # e.g., "STU_CS_2024_001" or "person_000"
    student_id = Column(String(50), nullable=True, index=True)  # Actual student/faculty ID
    department = Column(String(50), nullable=True, index=True)  # CS, IT, MECH, etc.
    user_type = Column(String(20), default='student')  # 'student', 'faculty', 'staff'
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    total_detections = Column(Integer, default=0)
    face_image_path = Column(String(255), nullable=True)
    detection_method = Column(String(50), default='appearance')  # 'face' or 'appearance'
    total_credits_earned = Column(Float, default=0.0)  # Cumulative blockchain credits
    
    # Relationship to events
    events = relationship('Event', back_populates='person', cascade='all, delete-orphan')
    activities = relationship('PersonActivity', back_populates='person', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Person(person_id='{self.person_id}', dept='{self.department}', credits={self.total_credits_earned})>"


class Event(Base):
    """Event detection table - Campus optimized"""
    __tablename__ = 'events'
    
    event_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    room_id = Column(String(50), index=True)  # e.g., "CS_LAB_101", "IT_CLASS_202"
    department = Column(String(50), nullable=True, index=True)  # Extracted from room_id
    
    # Detection details
    occupancy = Column(Boolean, default=False)
    person_count = Column(Integer, default=0)
    
    # Person reference (nullable for events without persons)
    person_id = Column(String(50), ForeignKey('persons.person_id'), nullable=True, index=True)
    
    # Bounding box data (stored as JSON)
    bbox = Column(JSON, nullable=True)  # {"x1": int, "y1": int, "x2": int, "y2": int}
    face_bbox = Column(JSON, nullable=True)
    
    # Detection metadata
    confidence = Column(Float, default=0.0)
    detection_method = Column(String(50), nullable=True)  # 'face' or 'appearance'
    
    # Device information
    devices_detected = Column(JSON, default=list)  # [{"type": "laptop", "confidence": 0.85}]
    device_count = Column(Integer, default=0)
    
    # Video metadata
    video_file = Column(String(255), nullable=True)
    frame_number = Column(Integer, nullable=True)
    
    # Action detection
    action_detected = Column(String(100), nullable=True)
    action_type = Column(String(50), nullable=True)  # 'sustainable', 'unsustainable', 'neutral'
    
    # Energy tracking
    energy_saved_estimate = Column(Float, default=0.0)  # Watts or kWh
    blockchain_credits = Column(Float, default=0.0)  # ₹ value
    
    # Device state tracking
    devices_on = Column(JSON, default=list)  # List of devices in ON state
    devices_off = Column(JSON, default=list)  # List of devices in OFF state
    lights_on = Column(Boolean, default=False)
    
    # Relationship to person
    person = relationship('Person', back_populates='events')
    
    def __repr__(self):
        return f"<Event(event_id={self.event_id}, person_id='{self.person_id}', timestamp='{self.timestamp}')>"
    
    def to_dict(self):
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'room_id': self.room_id,
            'occupancy': self.occupancy,
            'person_count': self.person_count,
            'person_id': self.person_id,
            'bbox': self.bbox,
            'face_bbox': self.face_bbox,
            'confidence': self.confidence,
            'detection_method': self.detection_method,
            'devices_detected': json.loads(self.devices_detected) if isinstance(self.devices_detected, str) else self.devices_detected,
            'device_count': self.device_count,
            'video_file': self.video_file,
            'frame_number': self.frame_number,
            'action_detected': self.action_detected,
            'action_type': self.action_type,
            'energy_saved_estimate': self.energy_saved_estimate,
            'blockchain_credits': self.blockchain_credits,
            'devices_on': json.loads(self.devices_on) if isinstance(self.devices_on, str) else self.devices_on,
            'devices_off': json.loads(self.devices_off) if isinstance(self.devices_off, str) else self.devices_off,
            'lights_on': self.lights_on
        }


class PersonActivity(Base):
    """Person activity log table for incentive tracking"""
    __tablename__ = 'person_activities'
    
    activity_id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(50), ForeignKey('persons.person_id'), index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    room_id = Column(String(50))
    
    activity_type = Column(String(50), index=True)  # 'presence', 'entry', 'exit', 'device_usage', 'violation'
    details = Column(JSON, nullable=True)  # Additional metadata
    
    # Incentive tracking
    incentive_points = Column(Integer, default=0)  # Positive or negative
    incentive_reason = Column(String(255), nullable=True)
    
    # Relationship to person
    person = relationship('Person', back_populates='activities')
    
    def __repr__(self):
        return f"<PersonActivity(activity_id={self.activity_id}, person_id='{self.person_id}', type='{self.activity_type}')>"
    
    def to_dict(self):
        """Convert activity to dictionary"""
        return {
            'activity_id': self.activity_id,
            'person_id': self.person_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'room_id': self.room_id,
            'activity_type': self.activity_type,
            'details': json.loads(self.details) if isinstance(self.details, str) else self.details,
            'incentive_points': self.incentive_points,
            'incentive_reason': self.incentive_reason
        }


class Database:
    """Database management class"""
    
    def __init__(self, db_url='sqlite:///outputs/sca_events.db'):
        """
        Initialize database connection
        
        Args:
            db_url: SQLAlchemy database URL
        """
        # Use SQLite options suited for multithreaded Flask dev server
        self.engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False},
            future=True
        )
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def close(self):
        """Close database connection"""
        self.Session.remove()
    
    def add_person(self, person_id, detection_method='appearance', face_image_path=None):
        """Add or update a person"""
        session = self.get_session()
        try:
            person = session.query(Person).filter_by(person_id=person_id).first()
            
            if person:
                # Update existing person
                person.last_seen = datetime.now()
                person.total_detections += 1
                if detection_method == 'face' and person.detection_method == 'appearance':
                    person.detection_method = 'face'
                if face_image_path:
                    person.face_image_path = face_image_path
            else:
                # Create new person
                person = Person(
                    person_id=person_id,
                    detection_method=detection_method,
                    face_image_path=face_image_path,
                    total_detections=1
                )
                session.add(person)
            
            session.commit()
            return person
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_event(self, event_data):
        """
        Add an event to the database
        
        Args:
            event_data: Dictionary with event information
        """
        session = self.get_session()
        try:
            event = Event(
                timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
                room_id=event_data.get('room_id'),
                occupancy=event_data.get('occupancy', False),
                person_count=event_data.get('person_count', 0),
                person_id=event_data.get('person_id'),
                bbox=json.dumps(event_data.get('bbox')) if event_data.get('bbox') else None,
                face_bbox=json.dumps(event_data.get('face_bbox')) if event_data.get('face_bbox') else None,
                confidence=event_data.get('confidence', 0.0),
                detection_method=event_data.get('detection_method'),
                devices_detected=json.dumps(event_data.get('devices_detected', [])),
                device_count=len(event_data.get('devices_detected', [])),
                video_file=event_data.get('video_file'),
                frame_number=event_data.get('frame_number'),
                action_detected=event_data.get('action_detected'),
                action_type=event_data.get('action_type'),
                energy_saved_estimate=event_data.get('energy_saved_estimate', 0.0),
                blockchain_credits=event_data.get('blockchain_credits', 0.0),
                devices_on=json.dumps(event_data.get('devices_on', [])),
                devices_off=json.dumps(event_data.get('devices_off', [])),
                lights_on=event_data.get('lights_on', False)
            )
            
            session.add(event)
            session.commit()
            return event
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_activity(self, person_id, activity_type, details=None, incentive_points=0, incentive_reason=None, room_id=None):
        """Add a person activity"""
        session = self.get_session()
        try:
            activity = PersonActivity(
                person_id=person_id,
                activity_type=activity_type,
                details=json.dumps(details) if details else None,
                incentive_points=incentive_points,
                incentive_reason=incentive_reason,
                room_id=room_id or 'CS_Lab_5'
            )
            
            session.add(activity)
            session.commit()
            return activity
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_all_persons(self):
        """Get all persons"""
        session = self.get_session()
        try:
            return session.query(Person).all()
        finally:
            session.close()
    
    def get_person_events(self, person_id):
        """Get all events for a person"""
        session = self.get_session()
        try:
            return session.query(Event).filter_by(person_id=person_id).all()
        finally:
            session.close()
    
    def get_recent_events(self, limit=100):
        """Get recent events"""
        session = self.get_session()
        try:
            return session.query(Event).order_by(Event.timestamp.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_person_score(self, person_id):
        """Calculate total incentive score for a person"""
        session = self.get_session()
        try:
            activities = session.query(PersonActivity).filter_by(person_id=person_id).all()
            total_score = sum(activity.incentive_points for activity in activities)
            return total_score
        finally:
            session.close()
    
    def get_leaderboard(self):
        """Get leaderboard with scores"""
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            scores = session.query(
                PersonActivity.person_id,
                func.sum(PersonActivity.incentive_points).label('total_score'),
                func.count(PersonActivity.activity_id).label('activity_count')
            ).group_by(PersonActivity.person_id).all()
            
            leaderboard = [
                {
                    'person_id': score[0],
                    'total_score': score[1] or 0,
                    'activity_count': score[2]
                }
                for score in scores
            ]
            
            leaderboard.sort(key=lambda x: x['total_score'], reverse=True)
            return leaderboard
        finally:
            session.close()


if __name__ == "__main__":
    # Test database creation when run directly
    print("Initializing database (standalone)...")
    db = Database()
    
    # Ensure outputs/face_database exists
    from pathlib import Path
    Path('outputs/face_database').mkdir(parents=True, exist_ok=True)

    # Add test person
    db.add_person('person_000', 'face', 'outputs/face_database/person_000_face.jpg')
    print("✓ Added test person")
    
    # Add test event
    test_event = {
        'timestamp': datetime.now().isoformat(),
        'room_id': 'CS_Lab_5',
        'occupancy': True,
        'person_count': 1,
        'person_id': 'person_000',
        'bbox': [100, 200, 300, 400],
        'confidence': 0.9,
        'detection_method': 'face',
        'devices_detected': [{'type': 'laptop', 'confidence': 0.85}],
        'video_file': 'test_video.mp4',
        'frame_number': 30
    }
    db.add_event(test_event)
    print("✓ Added test event")
    
    # Add test activity
    db.add_activity('person_000', 'presence', {'devices_nearby': 1}, incentive_points=1, incentive_reason='room_presence')
    print("✓ Added test activity")
    
    # Query data
    persons = db.get_all_persons()
    print(f"\n✓ Total persons in database: {len(persons)}")
    
    events = db.get_recent_events(10)
    print(f"✓ Recent events: {len(events)}")
    
    leaderboard = db.get_leaderboard()
    print(f"✓ Leaderboard entries: {len(leaderboard)}")
    
    print("\n✓ Database initialized successfully!")
    print("Database file: outputs/sca_events.db")
