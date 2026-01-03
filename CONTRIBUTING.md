# Contributing to SCA CV Module

Thank you for considering contributing to the Sustainable Campus Automation Computer Vision Module! 

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sca-cv-module.git
   cd sca-cv-module
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\Activate.ps1  # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install opencv-python opencv-contrib-python numpy flask ultralytics sqlalchemy
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Use meaningful variable names
   - Add docstrings to functions
   - Comment complex logic
   - Follow PEP 8 style guide

3. **Test your changes**:
   ```bash
   python app.py  # Test API
   python query_database.py  # Test database queries
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

## Commit Message Guidelines

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `perf:` Performance improvements

## Code Standards

- **Python**: Follow PEP 8
- **Type Hints**: Use where applicable
- **Documentation**: Update README.md for new features
- **Performance**: Maintain <31ms latency for real-time processing

## Adding New Devices

```python
# In energy_analyzer.py
DEVICE_POWER = {
    'laptop': 45,
    'your_device': POWER_IN_WATTS  # Add here
}

# In cv_processor.py
device_classes = {
    62: 'laptop',
    XX: 'your_device'  # Add YOLO class ID
}
```

## Testing

Before submitting a PR:
- [ ] Code runs without errors
- [ ] Performance metrics maintained (>90% accuracy)
- [ ] Documentation updated
- [ ] No unnecessary dependencies added

## Questions?

Open an issue or start a discussion on GitHub!
