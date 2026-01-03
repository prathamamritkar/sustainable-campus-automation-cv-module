#!/usr/bin/env python3
"""
Quick setup script for SCA CV Module
Initializes directories and database
"""
import os
from pathlib import Path

def setup():
    """Setup project directories and database"""
    print("🚀 Setting up SCA CV Module...")
    
    # Create directories
    dirs = ['models', 'uploads', 'outputs', 'outputs/face_database']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Initialize database
    try:
        from database import Database
        db = Database()
        print("✓ Database initialized: outputs/sca_events.db")
    except Exception as e:
        print(f"⚠ Database initialization failed: {e}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    required = ['cv2', 'numpy', 'flask', 'ultralytics', 'sqlalchemy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"✗ {pkg} - MISSING")
    
    if missing:
        print(f"\n⚠ Install missing packages:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n✅ All dependencies installed!")
    
    print("\n🎉 Setup complete! Run: python app.py")

if __name__ == "__main__":
    setup()
