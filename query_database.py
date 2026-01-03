"""
Database Query Examples - SCA CV Module
Demonstrates how to query the SQLite database
"""
from database import Database, Event, Person, PersonActivity
from datetime import datetime, timedelta

# Initialize database connection
db = Database()

print("=" * 60)
print("SCA CV MODULE - DATABASE QUERIES")
print("=" * 60)

# 1. Get all persons
print("\n1. ALL PERSONS IN DATABASE")
print("-" * 60)
persons = db.get_all_persons()
for person in persons:
    print(f"  {person.person_id}: {person.total_detections} detections, "
          f"Method: {person.detection_method}, "
          f"Last seen: {person.last_seen}")

# 2. Get recent events
print("\n2. RECENT EVENTS (Last 10)")
print("-" * 60)
recent_events = db.get_recent_events(limit=10)
for event in recent_events:
    devices = event.device_count
    print(f"  Event #{event.event_id}: {event.person_id or 'No person'} "
          f"in {event.room_id}, {devices} devices, "
          f"Frame: {event.frame_number}")

# 3. Get events for a specific person
print("\n3. EVENTS FOR person_000")
print("-" * 60)
if persons:
    person_id = persons[0].person_id
    person_events = db.get_person_events(person_id)
    print(f"  Total events for {person_id}: {len(person_events)}")
    for event in person_events[:5]:  # Show first 5
        print(f"    - Frame {event.frame_number}: "
              f"Confidence {event.confidence}, "
              f"Method: {event.detection_method}")

# 4. Get leaderboard
print("\n4. INCENTIVE LEADERBOARD")
print("-" * 60)
leaderboard = db.get_leaderboard()
print(f"  {'Rank':<6} {'Person ID':<15} {'Score':<10} {'Activities'}")
print("  " + "-" * 50)
for i, entry in enumerate(leaderboard[:10], 1):
    print(f"  {i:<6} {entry['person_id']:<15} {entry['total_score']:<10} {entry['activity_count']}")

# 5. Get person score
print("\n5. DETAILED PERSON SCORE")
print("-" * 60)
if persons:
    person_id = persons[0].person_id
    score = db.get_person_score(person_id)
    print(f"  {person_id} total score: {score} points")

# 6. Get database statistics
print("\n6. DATABASE STATISTICS")
print("-" * 60)
session = db.get_session()

total_persons = session.query(Person).count()
total_events = session.query(Event).count()
total_activities = session.query(PersonActivity).count()

print(f"  Total Persons: {total_persons}")
print(f"  Total Events: {total_events}")
print(f"  Total Activities: {total_activities}")
print(f"  Database file: outputs/sca_events.db")

session.close()

# 7. Query by time range
print("\n7. EVENTS FROM LAST HOUR")
print("-" * 60)
session = db.get_session()
one_hour_ago = datetime.now() - timedelta(hours=1)
recent_events_hour = session.query(Event).filter(Event.timestamp >= one_hour_ago).all()
print(f"  Events in last hour: {len(recent_events_hour)}")
session.close()

# 8. Query by room
print("\n8. EVENTS BY ROOM")
print("-" * 60)
session = db.get_session()
room_events = session.query(Event).filter_by(room_id='CS_Lab_5').all()
print(f"  Events in CS_Lab_5: {len(room_events)}")
session.close()

print("\n" + "=" * 60)
print("Database queries completed!")
print("=" * 60)

# Close database connection
db.close()
