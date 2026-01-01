# 📁 Project Structure - SEMIINTEL

**Organized on January 1, 2026**

## Root Directory

```
SemiIntel/
├── 📄 README.md                    # Quick start guide (points to docs/)
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 packages.txt                 # System dependencies for deployment
├── 🐍 app.py                       # Main Streamlit web application
├── 🐍 main.py                      # CLI interface
├── 🐍 demo.py                      # Demo scripts
│
├── 📁 docs/                        # 📚 All Documentation
│   ├── README.md                   # Complete project documentation
│   ├── DEPLOYMENT.md               # Deployment guide (Streamlit, Azure, etc.)
│   ├── WEB_APP_README.md          # Web application features
│   ├── WEB_APP_SUCCESS.md         # Web app implementation notes
│   ├── PROJECT_SUMMARY.md         # Executive summary
│   ├── PROJECT_STRUCTURE.txt      # Project file listing
│   ├── ML_NLP_FEATURES.md         # ML/NLP capabilities
│   ├── ML_METHODS_FIXED.md        # ML implementation fixes
│   ├── CLASSIFIER_FIXES_FINAL.md  # Classifier improvements
│   ├── CV_FOLD_FIX.md             # Cross-validation fixes
│   ├── FIXES_APPLIED.md           # General fixes log
│   ├── RUN_COMMANDS.md            # Command reference
│   └── INDEX.md                   # Documentation index
│
├── 📁 scripts/                     # 🛠️ All Scripts
│   ├── setup_github.ps1           # Automated GitHub setup
│   ├── run_web.ps1                # Run web app (PowerShell)
│   ├── run_web.bat                # Run web app (Batch)
│   └── QUICK_REFERENCE.sh         # Command reference (Bash)
│
├── 📁 modules/                     # 🐍 Python Modules
│   ├── __init__.py
│   ├── dataset_loader.py          # Kaggle dataset management
│   ├── dorking_engine.py          # Google Dorking queries
│   ├── github_scanner.py          # GitHub/StackOverflow scanner
│   ├── ml_analyzer.py             # ML models & pipeline
│   ├── nlp_analyzer.py            # NLP analysis tools
│   └── pdf_parser.py              # PDF extraction
│
├── 📁 data/                        # 📊 Data Files
│   ├── kaggle_datasets/           # Kaggle datasets
│   └── raw_datasheets/            # PDF datasheets
│
├── 📁 .streamlit/                  # ⚙️ Streamlit Configuration
│   └── config.toml                # Theme & server settings
│
└── 📁 .github/                     # 🔧 GitHub Configuration
    └── workflows/
        └── azure-webapps-python.yml  # Azure deployment workflow

```

## Clean Root Directory Benefits

✅ **Organized** - All docs in `docs/`, all scripts in `scripts/`
✅ **Professional** - Clean root directory structure
✅ **Maintainable** - Easy to find files by category
✅ **Scalable** - Clear organization for future additions
✅ **Deployable** - Essential files only in root

## Quick Access

### Documentation
```powershell
# View all documentation
ls docs/

# Read main documentation
cat docs/README.md

# View deployment guide
cat docs/DEPLOYMENT.md
```

### Scripts
```powershell
# List all scripts
ls scripts/

# Run web application
.\scripts\run_web.ps1

# Setup GitHub
.\scripts\setup_github.ps1
```

### Application
```powershell
# Run web app
python -m streamlit run app.py

# Run CLI
python main.py
```

## File Count Summary

- **Root:** 7 essential files (README, LICENSE, requirements, etc.)
- **docs/:** 13 documentation files
- **scripts/:** 4 executable scripts
- **modules/:** 7 Python modules
- **data/:** Dataset storage
- **.streamlit/:** 1 config file
- **.github/:** CI/CD workflows

## Updated References

All internal references have been updated:
- ✅ `app.py` - Updated to read from `scripts/` and `docs/`
- ✅ `README.md` - New quick start guide with links to docs
- ✅ Deployment page - Points to correct file locations

---

**Structure organized for:**
- Professional presentation
- Easy navigation
- Clear documentation
- Simple deployment
- Maintainable codebase
