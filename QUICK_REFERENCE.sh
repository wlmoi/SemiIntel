#!/usr/bin/env python3
"""
SEMIINTEL QUICK REFERENCE - Command Line Examples

Copy and paste these commands to run SEMIINTEL analyses.
"""

# ============================================================================
# PHASE 1: OSINT INTELLIGENCE GATHERING
# ============================================================================

# 1.1 - Generate Google Dorking Queries
python main.py --dorking STM32F407VG STM32H7 \
               --doc-types datasheet errata reference_manual \
               --output dorking_queries.txt

# 1.2 - Parse PDFs and extract metadata
python main.py --pdf ./data/raw_datasheets

# 1.3 - Analyze community issues
python main.py --community STM32F407VG STM32H7 STM32L4 \
               --report community_analysis.json

# ============================================================================
# PHASE 2: MACHINE LEARNING ANALYSIS
# ============================================================================

# 2.1 - Run ML analysis on specific chips
python main.py --ml STM32F407VG STM32H7 \
               --report ml_predictions.json

# 2.2 - Load and inspect Kaggle datasets
python main.py --datasets

# 2.3 - Demo ML pipeline
python demo.py --ml

# ============================================================================
# PHASE 3: NATURAL LANGUAGE PROCESSING
# ============================================================================

# 3.1 - Run NLP analysis on technical documents
python main.py --nlp STM32F407VG STM32H7 \
               --report nlp_analysis.json

# 3.2 - Demo NLP capabilities
python demo.py --nlp

# ============================================================================
# COMPLETE PIPELINES
# ============================================================================

# 4.1 - Full analysis (OSINT + ML + NLP)
python main.py --all STM32F407VG STM32H7 STM32H7 STM32L4 \
               --output queries.txt \
               --report complete_analysis.json

# 4.2 - Complete demonstration (ML + NLP)
python demo.py --full

# 4.3 - List all available chip models
python main.py --list-chips

# ============================================================================
# INDIVIDUAL MODULE USAGE IN PYTHON
# ============================================================================

# DORKING ENGINE
from modules.dorking_engine import DorkingEngine

engine = DorkingEngine()
query = engine.generate_dork_query("STM32F407VG", "datasheet", "st.com")
print(engine.format_for_google(query))

# PDF PARSER
from modules.pdf_parser import PDFMetadataExtractor

parser = PDFMetadataExtractor()
metadata = parser.extract_metadata("datasheet.pdf")
print(metadata["emails"])

# GITHUB SCANNER
from modules.github_scanner import GitHubScanner

scanner = GitHubScanner()
issues = scanner.search_repositories("STM32F407VG")
for issue in issues[:3]:
    print(f"{issue.title} [{issue.severity}]")

# SEVERITY CLASSIFIER (ML)
from modules.ml_analyzer import SeverityClassifier

classifier = SeverityClassifier()
classifier.train()
pred = classifier.predict("UART transmission crashes system")
print(f"Severity: {pred.predicted_value} ({pred.confidence:.1%})")

# NLP ANALYSIS
from modules.nlp_analyzer import NLPAnalyzer

analyzer = NLPAnalyzer()
doc_analysis = analyzer.analyze_document(datasheet_text)
print(f"Keywords: {doc_analysis['keywords']}")
print(f"Entities: {len(doc_analysis['entities'])}")

# DATASET LOADING
from modules.dataset_loader import DatasetManager

manager = DatasetManager()
issues_df = manager.load_dataset("github_issues")
print(f"Loaded {len(issues_df)} issues")

# ============================================================================
# OUTPUT FILES GENERATED
# ============================================================================

# After running analyses, check for these output files:
# 
# dorking_queries.txt          - Google search queries
# extracted_meta.csv           - PDF metadata (CSV)
# semiintel_report.json        - Complete JSON report
# ml_analysis.json             - ML predictions and metrics
# nlp_analysis.json            - NLP results (entities, keywords, sentiment)
# community_analysis.json      - GitHub/SO analysis
#
# All reports include timestamp and detailed metrics

# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

# Workflow 1: Quick Intelligence Gathering
# ─────────────────────────────────────────
echo "Gathering OSINT..."
python main.py --dorking STM32F407VG --output queries.txt
python main.py --community STM32F407VG --report osint_report.json
# Result: Search queries + community issues found

# Workflow 2: ML-Driven Verification Planning
# ────────────────────────────────────────────
echo "Planning verification..."
python main.py --ml STM32F407VG --report verification_plan.json
# Result: Issue severity predictions + clustering

# Workflow 3: Document Intelligence
# ──────────────────────────────────
echo "Analyzing technical docs..."
python main.py --nlp STM32F407VG --report doc_analysis.json
# Result: Extracted specs, keywords, entities

# Workflow 4: Complete Analysis (for interview prep)
# ───────────────────────────────────────────────────
echo "Running complete analysis..."
python main.py --all STM32F407VG STM32H7 --report interview_ready.json
# Result: Everything in one report

# ============================================================================
# DEBUGGING & VERBOSE OUTPUT
# ============================================================================

# Run with verbose output for debugging
python main.py --all STM32F407VG --verbose

# Run individual demos for testing
python demo.py --ml         # Test ML
python demo.py --nlp        # Test NLP
python demo.py --datasets   # Test dataset loading

# ============================================================================
# PERFORMANCE NOTES
# ============================================================================

# Dorking:              ~0.5s for 500 queries
# PDF Parsing:          ~5-10s for 50 PDFs
# Community Analysis:   ~3s for 10 chips
# ML Training:          ~5-10s (100+ issues)
# ML Predictions:       <10ms each
# NLP Analysis:         ~20-50ms per document
# Total Pipeline:       ~30-60s end-to-end

# ============================================================================
# TIPS FOR STMicroelectronics INTERVIEW
# ============================================================================

# 1. Show this working during technical interview:
#    python demo.py --full
#    Demonstrates all capabilities in 60 seconds

# 2. Explain what each phase does:
#    - "Phase 1 automates document discovery..."
#    - "Phase 2 uses ML to prioritize testing..."
#    - "Phase 3 extracts specifications automatically..."

# 3. Highlight custom thinking:
#    - Issue clustering identifies bug categories
#    - Severity classifier prioritizes critical issues
#    - NER extracts requirements from PDFs

# 4. Discuss production readiness:
#    - Modular design for maintainability
#    - Cross-validation for model reliability
#    - 10 production-grade Kaggle datasets

# ============================================================================
# NEXT STEPS
# ============================================================================

# After demonstrating SEMIINTEL:
# ✓ Explain your verification methodology
# ✓ Discuss how ML improves test planning
# ✓ Show understanding of IC design challenges
# ✓ Ask about STMicroelectronics' verification tools
# ✓ Mention interest in extending SEMIINTEL in role

# Remember: "I built this tool because I'm genuinely interested in how 
# STMicroelectronics ensures their products work reliably. Rather than just 
# learning verification concepts, I created a practical tool that applies 
# OSINT, ML, and NLP to real semiconductor challenges."

# ============================================================================
# USEFUL KAGGLE DATASETS FOR EXTENSION
# ============================================================================

# Once you connect to Kaggle API, these datasets are available:
# 1. GitHub Issues Archive
# 2. Stack Overflow Questions  
# 3. IC Performance Benchmarks
# 4. Semiconductor Manufacturing Data
# 5. IoT Device Failure Logs
# 6. Hardware Bug Reports
# 7. Technical Documentation Corpus
# 8. Electronics Product Reviews
# 9. Microcontroller Specifications
# 10. Community Bug Tracker Dataset

# See dataset_loader.py for full registry and loading code.
