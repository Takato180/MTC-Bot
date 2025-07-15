# Contributing to MTC-Bot

Thank you for your interest in contributing to MTC-Bot! This document provides guidelines and instructions for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Poetry for dependency management
- Docker and Docker Compose
- Git

### Development Setup

1. **Fork the repository**
   ```bash
   git fork https://github.com/Takato180/MTC-Bot.git
   cd MTC-Bot
   ```

2. **Install development dependencies**
   ```bash
   poetry install --with dev
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to ensure everything works**
   ```bash
   python -m pytest
   ```

## 🏗️ Development Workflow

### Branch Management

- `main`: Production-ready code
- `develop`: Development branch for integration
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes for production

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Code Style**: Follow PEP 8 and use our linting tools
2. **Documentation**: Update docstrings and README if needed
3. **Tests**: Add tests for new functionality
4. **Commits**: Use clear, descriptive commit messages

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Example:
```
feat(patchtst): add hyperparameter optimization

- Implement Optuna-based hyperparameter search
- Add configuration options for optimization
- Update training script to support optimization mode

Closes #123
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_patchtst.py

# Run with coverage
python -m pytest --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test system performance

### Writing Tests

```python
import pytest
from src.strategy_service.patchtst.model import PatchTST, PatchTSTConfig

def test_patchtst_model_creation():
    \"\"\"Test PatchTST model creation with valid config.\"\"\"
    config = PatchTSTConfig(seq_len=100, pred_len=20, n_vars=5)
    model = PatchTST(config)
    
    assert model.seq_len == 100
    assert model.pred_len == 20
    assert model.n_vars == 5
```

## 📝 Code Quality

### Linting and Formatting

We use `ruff` for linting and formatting:

```bash
# Format code
ruff format src/

# Check linting
ruff check src/

# Fix linting issues
ruff check --fix src/
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy src/
```

### Pre-commit Hooks

Our pre-commit hooks run:
- `ruff` formatting and linting
- `mypy` type checking
- `pytest` on changed files

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def train_model(data: np.ndarray, epochs: int = 100) -> dict:
    \"\"\"Train the PatchTST model.
    
    Args:
        data: Training data array of shape (n_samples, n_features)
        epochs: Number of training epochs
    
    Returns:
        Dictionary containing training metrics and model state
    
    Raises:
        ValueError: If data shape is invalid
        RuntimeError: If training fails
    \"\"\"
```

### API Documentation

Update API documentation when adding new public methods or classes.

## 🐛 Bug Reports

### Before Reporting

1. Check if the issue already exists
2. Try to reproduce the issue
3. Gather relevant information

### Bug Report Template

```markdown
## Bug Description
A clear description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.12.0]
- PyTorch Version: [e.g., 2.1.0]
- GPU: [if applicable]

## Additional Context
Any other relevant information.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
A clear description of the requested feature.

## Motivation
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives
Any alternative solutions considered?

## Implementation Notes
Technical considerations or constraints.
```

## 🔧 Development Guidelines

### Code Organization

- Follow the existing project structure
- Keep modules focused and cohesive
- Use meaningful names for functions and variables
- Add type hints to all public functions

### Performance Considerations

- Profile code for performance bottlenecks
- Use appropriate data structures
- Consider memory usage for large datasets
- Optimize GPU usage when applicable

### Security Guidelines

- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all inputs
- Follow security best practices

## 🎯 Areas for Contribution

### High Priority

- [ ] Model performance improvements
- [ ] Additional exchange integrations
- [ ] Enhanced risk management features
- [ ] Documentation improvements

### Medium Priority

- [ ] UI/UX enhancements
- [ ] Additional technical indicators
- [ ] Performance optimizations
- [ ] Test coverage improvements

### Low Priority

- [ ] Code refactoring
- [ ] Minor bug fixes
- [ ] Documentation updates
- [ ] Example improvements

## 📋 Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Update documentation
3. Add/update tests for new functionality
4. Follow the code style guidelines
5. Rebase your branch on the latest develop

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: Additional testing if needed
4. **Approval**: Maintainer approves the PR
5. **Merge**: PR is merged into develop branch

## 🏆 Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributors page
- Special recognition for significant contributions

## 📞 Getting Help

- **Email**: masymyt@gmail.com
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions

## 📜 Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all interactions.

## 🙏 Thank You

We appreciate all contributions, whether they're bug fixes, feature additions, documentation improvements, or just feedback. Every contribution helps make MTC-Bot better!

---

Happy coding! 🚀