# 🚀 Git Setup & Push to GitHub

## Quick Start

```powershell
# 1. Initialize Git repository
git init

# 2. Add all files
git add .

# 3. Create initial commit
git commit -m "feat: Initial commit - SCA CV Module v1.0

- Computer vision module with 93% accuracy
- 3 optimization modes (precision/balanced/recall)
- Person recognition with Kalman filtering
- Device state detection with ensemble methods
- Energy analytics with blockchain credits
- SQLite database with campus tracking
- Flask REST API with 11 endpoints
- Comprehensive documentation"

# 4. Create GitHub repository (via GitHub CLI or web interface)
# Option A: Using GitHub CLI (if installed)
gh repo create sca-cv-module --public --source=. --remote=origin

# Option B: Or create manually on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/sca-cv-module.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

## Using GitHub CLI (Recommended)

```powershell
# Install GitHub CLI (if not installed)
winget install GitHub.cli

# Login to GitHub
gh auth login

# Create repository and push
gh repo create sca-cv-module --public --source=. --remote=origin --push
```

## Manual GitHub Setup

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `sca-cv-module`
3. **Description**: `Real-time energy monitoring system with 93% accuracy for sustainable campus automation`
4. **Public/Private**: Choose based on preference
5. **Don't initialize** with README (we have one)
6. **Click**: Create repository
7. **Run**:
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/sca-cv-module.git
   git branch -M main
   git push -u origin main
   ```

## Repository Settings

After pushing, configure:

### About Section
- **Description**: Real-time energy monitoring with computer vision for sustainable campuses
- **Website**: (optional)
- **Topics**: `computer-vision`, `sustainability`, `energy-monitoring`, `yolov8`, `opencv`, `python`, `flask`, `campus-automation`, `blockchain`, `green-tech`

### Branch Protection (Optional)
- Go to Settings → Branches → Add rule
- Branch name: `main`
- ☑️ Require pull request reviews before merging
- ☑️ Require status checks to pass

## Next Steps

```powershell
# Clone your repository
git clone https://github.com/YOUR_USERNAME/sca-cv-module.git
cd sca-cv-module

# Setup virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from database import Database; Database()"

# Run API server
python app.py
```

## Repository Structure

```
sca-cv-module/
├── .github/               # GitHub templates
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── FUNDING.yml
├── models/                # YOLO weights (auto-download)
├── outputs/               # Event logs & database
├── uploads/               # Video uploads
├── .gitignore            # Git ignore rules
├── app.py                # Flask REST API
├── CONTRIBUTING.md       # Contribution guidelines
├── cv_processor.py       # CV detection module
├── database.py           # SQLite ORM
├── energy_analyzer.py    # Energy calculation
├── incentive_tracker.py  # Blockchain credits
├── LICENSE               # MIT License
├── query_database.py     # Database utilities
├── README.md             # Comprehensive docs
└── requirements.txt      # Python dependencies
```

## GitHub Features

### Enable GitHub Actions (Optional)
Create `.github/workflows/test.yml` for automated testing.

### Add GitHub Pages (Optional)
Enable Pages from Settings → Pages for documentation hosting.

### Add Topics
Settings → Topics: `computer-vision`, `yolov8`, `sustainability`, `flask-api`

---

**Ready to push! 🚀**
