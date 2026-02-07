# Contributing to Medical Lesion Detection

Thank you for your interest in contributing to the Medical Lesion Detection project! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please read and abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive criticism
- Respect confidentiality of others
- Report unacceptable behavior

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (recommended)

### Fork & Clone

```bash
# 1. Fork the repository on GitHub
# Click "Fork" button on the repository page

# 2. Clone your fork
git clone https://github.com/dinhtuandev/medical-lesion-detection.git
cd medical-lesion-detection

# 3. Add upstream remote
git remote add upstream https://github.com/dinhtuandev/medical-lesion-detection.git
```

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests to ensure setup works
pytest tests/ -v
```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
# Fetch latest changes from upstream
git fetch upstream

# Create new branch from upstream/main
git checkout -b feature/your-feature-name upstream/main
```

### 2. Make Changes

```bash
# Edit files
# Test your changes frequently
pytest tests/ -v

# Check code style
flake8 src/
black --check src/

# Format code if needed
black src/
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocess.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Update Documentation

- Update README.md if adding features
- Update docstrings in code
- Add comments for complex logic
- Create/update relevant guide files

---

## Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code formatting (no functional changes)
- **refactor**: Code refactoring
- **test**: Adding/modifying tests
- **chore**: Build, dependency updates

### Subject

- Use imperative mood ("add feature" not "added feature")
- Don't capitalize first letter
- No period at the end
- Maximum 50 characters

### Body

- Explain what and why, not how
- Wrap at 72 characters
- Separate from subject with blank line

### Examples

```
feat(preprocessing): add CLAHE adaptive tile size

- Allow user to configure CLAHE tile size
- Add validation for tile size range
- Update documentation

Closes #123
```

```
fix(model): correct bounding box coordinates

The coordinates were being calculated incorrectly
for non-square images. Fixed by normalizing input
dimensions before calculation.

Fixes #456
```

---

## Pull Request Process

### Before Submitting

1. **Update your branch**

```bash
git fetch upstream
git rebase upstream/main
```

2. **Run all checks**

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
flake8 src/ app.py
black src/ app.py
```

3. **Push your changes**

```bash
git push origin feature/your-feature-name
```

### PR Description Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing

- [ ] Tests added/modified
- [ ] All tests passing
- [ ] Coverage maintained/improved

## Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Commits follow conventions

## Related Issues

Closes #issue_number
```

### Review Process

1. At least one review required
2. All CI checks must pass
3. Discussions resolved
4. Maintainer approves
5. Squash and merge (if requested)

---

## Reporting Issues

### Bug Reports

Include:

```
- Python version
- OS (Windows/Mac/Linux)
- Error message/traceback
- Steps to reproduce
- Expected vs actual behavior
```

### Feature Requests

Describe:

```
- Use case/motivation
- Proposed solution
- Alternative solutions
- Additional context
```

### Questions

- Check existing issues/docs first
- Use Discussion tab if available
- Be specific and provide examples

### Issue Template Example

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What should happen

**Actual behavior**
What actually happens

**Environment**

- OS: [e.g. Windows 10]
- Python: [e.g. 3.10]
- Version: [e.g. v1.0.0]

**Additional context**
Any additional context
```

---

## Documentation

### Guidelines

- Write in clear, concise English
- Use examples for complex concepts
- Keep docs up-to-date
- Add docstrings to functions
- Document breaking changes
- Include code examples

### Docstring Format

```python
def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE enhancement to image.

    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    enhances contrast locally without noise amplification.

    Args:
        image (ndarray): Input BGR image
        clip_limit (float): Contrast limit, default 2.0
        tile_size (int): Tile grid size, default 8

    Returns:
        ndarray: Enhanced BGR image

    Raises:
        TypeError: If image is not ndarray
        ValueError: If clip_limit or tile_size invalid

    Example:
        >>> img = cv2.imread('xray.jpg')
        >>> enhanced = apply_clahe(img, clip_limit=2.0)
    """
```

---

## Development Tips

### Testing

- Write tests for new features
- Fix failing tests before PR
- Aim for >90% coverage
- Test edge cases

### Code Quality

- Follow PEP 8 style guide
- Use type hints
- Keep functions small
- Add meaningful comments
- Avoid code duplication

### Performance

- Profile code before optimizing
- Document performance improvements
- Benchmark on realistic data
- Consider memory usage

### Security

- Validate user inputs
- Don't hardcode secrets
- Use safe APIs
- Check dependencies vulnerabilities

---

## Questions?

- 📖 Check [README.md](README.md) for usage
- 💻 See [CODE_EXPLANATION.md](CODE_EXPLANATION.md) for code details
- 🧪 Read [TESTING_GUIDE.md](TESTING_GUIDE.md) for testing
- 📊 Review [TEST_RESULTS_REPORT.md](TEST_RESULTS_REPORT.md) for metrics

---

## Recognition

Contributors will be recognized in:

- Repository README
- Release notes
- Contributor list

Thank you for contributing! 🎉
