from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from cv_processor import CVProcessor
import json
from datetime import datetime, timedelta
from database import Database, Event, Person, PersonActivity
from energy_analyzer import EnergyAnalyzer

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
MODELS_FOLDER = Path('models')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create folders if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize CV processor and database lazily
cv_processor = None
db = None
energy_analyzer = None


@app.before_first_request
def initialize_resources():
    """Ensure folders exist and initialize database and analytics on first run."""
    global db, energy_analyzer, cv_processor

    # Ensure essential directories exist
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    (OUTPUT_FOLDER / 'face_database').mkdir(parents=True, exist_ok=True)

    # Initialize database and analytics
    if db is None:
        db = Database()

    if energy_analyzer is None:
        energy_analyzer = EnergyAnalyzer()

    # CV processor will be created on first use via get_cv_processor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_cv_processor():
    """Get or create CV processor instance with database enabled"""
    global cv_processor
    if cv_processor is None:
        model_path = MODELS_FOLDER / 'yolov8n.pt'
        mp = str(model_path) if model_path.exists() else None
        cv_processor = CVProcessor(use_database=True, model_path=mp)
    return cv_processor


@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        'service': 'SCA CV Module API',
        'version': '2.0',
        'database': 'SQLite with SQLAlchemy',
        'endpoints': {
            'POST /upload': 'Upload a video file',
            'POST /process': 'Process an uploaded video',
            'GET /results/<filename>': 'Get processing results',
            'GET /status': 'Get API status',
            'GET /db/events': 'Get all events from database',
            'GET /db/events/<event_id>': 'Get specific event',
            'GET /db/persons': 'Get all persons',
            'GET /db/persons/<person_id>': 'Get specific person details',
            'GET /db/persons/<person_id>/events': 'Get events for a person',
            'GET /db/persons/<person_id>/activities': 'Get activities for a person',
            'GET /db/leaderboard': 'Get person leaderboard with scores',
            'GET /db/stats': 'Get database statistics',
            'GET /energy/report': 'Get energy usage report',
            'GET /energy/blockchain-credits': 'Get blockchain credits summary',
            'GET /energy/sustainable-actions': 'Get sustainable actions log',
            'GET /energy/live-metrics': 'Get real-time energy metrics'
        }
    })


@app.route('/status', methods=['GET'])
def status():
    """Get API status"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'uploads_count': len(list(UPLOAD_FOLDER.glob('*'))),
        'outputs_count': len(list(OUTPUT_FOLDER.glob('*.json')))
    })


@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Upload a video file
    
    Form data:
        file: Video file
        
    Returns:
        JSON with upload status and filename
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_with_timestamp = f"{timestamp}_{filename}"
    filepath = UPLOAD_FOLDER / filename_with_timestamp
    
    try:
        file.save(str(filepath))
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename_with_timestamp,
            'path': str(filepath),
            'size_bytes': filepath.stat().st_size
        }), 201
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/process', methods=['POST'])
def process_video():
    """
    Process an uploaded video
    
    JSON body:
        filename: Name of uploaded video file
        confidence: (optional) Confidence threshold (default: 0.5)
        
    Returns:
        JSON with processing results
    """
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': 'filename required in request body'}), 400
    
    filename = data['filename']
    confidence = data.get('confidence', 0.5)
    
    # Validate confidence threshold
    try:
        confidence = float(confidence)
        if not 0 < confidence <= 1:
            raise ValueError()
    except:
        return jsonify({'error': 'confidence must be between 0 and 1'}), 400
    
    video_path = UPLOAD_FOLDER / filename
    
    if not video_path.exists():
        return jsonify({'error': f'Video file not found: {filename}'}), 404
    
    # Generate output filename
    output_filename = f"{video_path.stem}_detections.json"
    output_path = OUTPUT_FOLDER / output_filename
    
    try:
        processor = get_cv_processor()
        results = processor.process_video(
            video_path=str(video_path),
            output_json_path=str(output_path),
            confidence_threshold=confidence
        )
        
        # Add output file info to results
        results['output_file'] = output_filename
        results['download_url'] = f'/results/{output_filename}'
        
        return jsonify({
            'message': 'Processing complete',
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    """
    Get processing results JSON file
    
    Args:
        filename: Name of the results JSON file
        
    Returns:
        JSON file or JSON data
    """
    output_path = OUTPUT_FOLDER / filename
    
    if not output_path.exists():
        return jsonify({'error': f'Results file not found: {filename}'}), 404
    
    # Check if user wants to download or view
    download = request.args.get('download', 'false').lower() == 'true'
    
    if download:
        return send_file(str(output_path), as_attachment=True)
    else:
        with open(output_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)


@app.route('/list/uploads', methods=['GET'])
def list_uploads():
    """List all uploaded video files"""
    files = []
    for filepath in UPLOAD_FOLDER.glob('*'):
        if filepath.is_file():
            files.append({
                'filename': filepath.name,
                'size_bytes': filepath.stat().st_size,
                'uploaded_at': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            })
    
    return jsonify({
        'count': len(files),
        'files': sorted(files, key=lambda x: x['uploaded_at'], reverse=True)
    })


@app.route('/list/results', methods=['GET'])
def list_results():
    """List all processing results"""
    files = []
    for filepath in OUTPUT_FOLDER.glob('*.json'):
        if filepath.is_file():
            files.append({
                'filename': filepath.name,
                'size_bytes': filepath.stat().st_size,
                'created_at': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            })
    
    return jsonify({
        'count': len(files),
        'files': sorted(files, key=lambda x: x['created_at'], reverse=True)
    })


@app.route('/summary/<filename>', methods=['GET'])
def get_summary(filename):
    """
    Get detection summary for a results file
    
    Args:
        filename: Name of the results JSON file
        
    Returns:
        Summary statistics
    """
    output_path = OUTPUT_FOLDER / filename
    
    if not output_path.exists():
        return jsonify({'error': f'Results file not found: {filename}'}), 404
    
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Calculate class distribution
    class_counts = {}
    for event in data.get('events', []):
        class_name = event['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    summary = {
        'video_file': data.get('video_file'),
        'total_detections': data.get('total_detections', 0),
        'duration_seconds': data.get('duration_seconds', 0),
        'class_distribution': class_counts,
        'processed_at': data.get('processed_at')
    }
    
    return jsonify(summary)


# ============================================================
# DATABASE ENDPOINTS
# ============================================================

@app.route('/db/events', methods=['GET'])
def get_db_events():
    """Get all events from database with pagination"""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    person_id = request.args.get('person_id', None)
    
    try:
        session = db.get_session()
        
        query = session.query(Event)
        
        if person_id:
            query = query.filter_by(person_id=person_id)
        
        query = query.order_by(Event.timestamp.desc())
        
        total = query.count()
        events = query.offset(offset).limit(limit).all()
        
        return jsonify({
            'total': total,
            'limit': limit,
            'offset': offset,
            'events': [event.to_dict() for event in events]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/db/events/<int:event_id>', methods=['GET'])
def get_db_event(event_id):
    """Get specific event by ID"""
    try:
        session = db.get_session()
        event = session.query(Event).filter_by(event_id=event_id).first()
        
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        return jsonify(event.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/db/persons', methods=['GET'])
def get_db_persons():
    """Get all persons from database"""
    try:
        persons = db.get_all_persons()
        
        persons_data = []
        for person in persons:
            persons_data.append({
                'person_id': person.person_id,
                'first_seen': person.first_seen.isoformat() if person.first_seen else None,
                'last_seen': person.last_seen.isoformat() if person.last_seen else None,
                'total_detections': person.total_detections,
                'detection_method': person.detection_method,
                'face_image_path': person.face_image_path
            })
        
        return jsonify({
            'total': len(persons_data),
            'persons': persons_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/db/persons/<person_id>', methods=['GET'])
def get_db_person(person_id):
    """Get specific person details"""
    try:
        session = db.get_session()
        person = session.query(Person).filter_by(person_id=person_id).first()
        
        if not person:
            return jsonify({'error': 'Person not found'}), 404
        
        # Get score
        score = db.get_person_score(person_id)
        
        return jsonify({
            'person_id': person.person_id,
            'first_seen': person.first_seen.isoformat() if person.first_seen else None,
            'last_seen': person.last_seen.isoformat() if person.last_seen else None,
            'total_detections': person.total_detections,
            'detection_method': person.detection_method,
            'face_image_path': person.face_image_path,
            'total_score': score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/db/persons/<person_id>/events', methods=['GET'])
def get_person_db_events(person_id):
    """Get all events for a specific person"""
    try:
        events = db.get_person_events(person_id)
        
        return jsonify({
            'person_id': person_id,
            'total_events': len(events),
            'events': [event.to_dict() for event in events]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/db/persons/<person_id>/activities', methods=['GET'])
def get_person_activities(person_id):
    """Get all activities for a specific person"""
    try:
        session = db.get_session()
        activities = session.query(PersonActivity).filter_by(person_id=person_id).order_by(PersonActivity.timestamp.desc()).all()
        
        return jsonify({
            'person_id': person_id,
            'total_activities': len(activities),
            'activities': [activity.to_dict() for activity in activities]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/db/leaderboard', methods=['GET'])
def get_db_leaderboard():
    """Get person leaderboard with scores"""
    try:
        leaderboard = db.get_leaderboard()
        
        return jsonify({
            'total_persons': len(leaderboard),
            'leaderboard': leaderboard
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/db/stats', methods=['GET'])
def get_db_stats():
    """Get database statistics"""
    try:
        session = db.get_session()
        
        total_persons = session.query(Person).count()
        total_events = session.query(Event).count()
        total_activities = session.query(PersonActivity).count()
        
        # Recent activity
        recent_events = session.query(Event).order_by(Event.timestamp.desc()).limit(5).all()
        
        return jsonify({
            'total_persons': total_persons,
            'total_events': total_events,
            'total_activities': total_activities,
            'recent_events': [event.to_dict() for event in recent_events]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


# ============================================================
# ENERGY ANALYTICS ENDPOINTS
# ============================================================

@app.route('/energy/report', methods=['GET'])
def get_energy_report():
    """
    Get comprehensive energy usage report
    
    Query params:
        hours: Time period in hours (default: 24)
        room_id: Filter by room (optional)
    """
    try:
        hours = request.args.get('hours', 24, type=int)
        room_id = request.args.get('room_id', None)
        
        session = db.get_session()
        from database import Event
        
        # Get events from specified time period
        time_threshold = datetime.now() - timedelta(hours=hours)
        query = session.query(Event).filter(Event.timestamp >= time_threshold)
        
        if room_id:
            query = query.filter_by(room_id=room_id)
        
        events = query.all()
        
        # Convert to dict for analysis
        event_dicts = [e.to_dict() for e in events]
        
        # Generate report
        report = energy_analyzer.generate_energy_report(event_dicts, time_period_hours=hours)
        
        # Add room-specific data
        report['room_id'] = room_id or 'ALL'
        report['events_analyzed'] = len(events)
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/energy/blockchain-credits', methods=['GET'])
def get_blockchain_credits():
    """
    Get blockchain credits summary
    
    Query params:
        person_id: Filter by person (optional)
        hours: Time period in hours (default: 24)
    """
    try:
        person_id = request.args.get('person_id', None)
        hours = request.args.get('hours', 24, type=int)
        
        session = db.get_session()
        from database import Event
        from sqlalchemy import func
        
        time_threshold = datetime.now() - timedelta(hours=hours)
        query = session.query(Event).filter(Event.timestamp >= time_threshold)
        
        if person_id:
            query = query.filter_by(person_id=person_id)
        
        # Calculate totals
        total_credits = session.query(func.sum(Event.blockchain_credits)).filter(
            Event.timestamp >= time_threshold
        ).scalar() or 0
        
        total_energy_saved = session.query(func.sum(Event.energy_saved_estimate)).filter(
            Event.timestamp >= time_threshold
        ).scalar() or 0
        
        # Get top earners
        top_earners = session.query(
            Event.person_id,
            func.sum(Event.blockchain_credits).label('total_credits'),
            func.count(Event.event_id).label('action_count')
        ).filter(
            Event.timestamp >= time_threshold,
            Event.person_id.isnot(None)
        ).group_by(Event.person_id).order_by(
            func.sum(Event.blockchain_credits).desc()
        ).limit(10).all()
        
        top_earners_list = [
            {
                'person_id': earner[0],
                'total_credits': float(earner[1] or 0),
                'action_count': earner[2]
            }
            for earner in top_earners
        ]
        
        result = {
            'time_period_hours': hours,
            'total_blockchain_credits': round(float(total_credits), 2),
            'total_energy_saved_watts': round(float(total_energy_saved), 2),
            'credits_per_kwh': energy_analyzer.CREDIT_RATE_PER_KWH,
            'top_earners': top_earners_list,
            'projected_annual_value': round(float(total_credits) * (8760 / hours), 2)
        }
        
        if person_id:
            result['person_id'] = person_id
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/energy/sustainable-actions', methods=['GET'])
def get_sustainable_actions():
    """
    Get log of sustainable and unsustainable actions
    
    Query params:
        action_type: Filter by 'sustainable' or 'unsustainable' (optional)
        limit: Number of results (default: 50)
    """
    try:
        action_type = request.args.get('action_type', None)
        limit = request.args.get('limit', 50, type=int)
        
        session = db.get_session()
        from database import Event
        
        query = session.query(Event).filter(Event.action_type.isnot(None))
        
        if action_type:
            query = query.filter_by(action_type=action_type)
        
        actions = query.order_by(Event.timestamp.desc()).limit(limit).all()
        
        actions_data = []
        for action in actions:
            actions_data.append({
                'event_id': action.event_id,
                'timestamp': action.timestamp.isoformat(),
                'person_id': action.person_id,
                'room_id': action.room_id,
                'action_type': action.action_type,
                'action_detected': action.action_detected,
                'energy_saved_estimate': action.energy_saved_estimate,
                'blockchain_credits': action.blockchain_credits,
                'devices_on_count': len(json.loads(action.devices_on) if isinstance(action.devices_on, str) else action.devices_on or []),
                'devices_off_count': len(json.loads(action.devices_off) if isinstance(action.devices_off, str) else action.devices_off or [])
            })
        
        # Calculate summary
        sustainable_count = len([a for a in actions_data if a['action_type'] == 'sustainable'])
        unsustainable_count = len([a for a in actions_data if a['action_type'] == 'unsustainable'])
        
        return jsonify({
            'total_actions': len(actions_data),
            'sustainable_actions': sustainable_count,
            'unsustainable_actions': unsustainable_count,
            'filter_applied': action_type,
            'actions': actions_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


@app.route('/energy/live-metrics', methods=['GET'])
def get_live_metrics():
    """Get real-time energy metrics from most recent events"""
    try:
        session = db.get_session()
        from database import Event
        
        # Get most recent event
        latest_event = session.query(Event).order_by(Event.timestamp.desc()).first()
        
        if not latest_event:
            return jsonify({'error': 'No events found'}), 404
        
        # Get events from last 5 minutes for trending
        five_min_ago = datetime.now() - timedelta(minutes=5)
        recent_events = session.query(Event).filter(
            Event.timestamp >= five_min_ago
        ).all()
        
        # Calculate metrics
        total_credits_5min = sum([e.blockchain_credits or 0 for e in recent_events])
        total_energy_5min = sum([e.energy_saved_estimate or 0 for e in recent_events])
        
        devices_on = json.loads(latest_event.devices_on) if isinstance(latest_event.devices_on, str) else latest_event.devices_on or []
        devices_off = json.loads(latest_event.devices_off) if isinstance(latest_event.devices_off, str) else latest_event.devices_off or []
        
        return jsonify({
            'current_time': datetime.now().isoformat(),
            'latest_event': {
                'timestamp': latest_event.timestamp.isoformat(),
                'room_id': latest_event.room_id,
                'occupancy': latest_event.occupancy,
                'person_count': latest_event.person_count,
                'devices_on_count': len(devices_on),
                'devices_off_count': len(devices_off),
                'lights_on': latest_event.lights_on,
                'action_type': latest_event.action_type,
                'blockchain_credits': latest_event.blockchain_credits
            },
            'last_5_minutes': {
                'total_events': len(recent_events),
                'total_credits_earned': round(total_credits_5min, 2),
                'total_energy_saved_watts': round(total_energy_5min, 2)
            },
            'device_power_reference': energy_analyzer.DEVICE_POWER,
            'credit_rate_per_kwh': energy_analyzer.CREDIT_RATE_PER_KWH
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()


if __name__ == '__main__':
    print("=" * 50)
    print("SCA CV Module API Server")
    print("=" * 50)
    print(f"Upload folder: {UPLOAD_FOLDER.absolute()}")
    print(f"Output folder: {OUTPUT_FOLDER.absolute()}")
    print(f"Models folder: {MODELS_FOLDER.absolute()}")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
