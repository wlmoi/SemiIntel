"""
SEMIINTEL - Semiconductor Intelligence Automation Tool
Main Entry Point

A Python-based Open Source Intelligence (OSINT) tool that automates the gathering
of intelligence on semiconductor components, specifically STMicroelectronics datasheets,
errata sheets, and technical documentation.

This tool demonstrates data parsing, automation, and verification skills essential
for IC Digital Design and Verification roles.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Import custom modules
from modules.dorking_engine import DorkingEngine, STM_CHIP_MODELS
from modules.pdf_parser import PDFMetadataExtractor, EmailIntelligenceExtractor
from modules.github_scanner import GitHubScanner, StackOverflowScanner, VerificationAnalyzer
from modules.ml_analyzer import MLPipeline, SeverityClassifier, IssueClusterer
from modules.dataset_loader import DatasetManager, KaggleDatasetRegistry, SyntheticDataGenerator
from modules.nlp_analyzer import NLPAnalyzer, NamedEntityRecognizer


class SemiIntelCLI:
    """
    Command-line interface for the SEMIINTEL tool.
    
    Orchestrates all modules to automate technical intelligence gathering.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.dorking_engine = DorkingEngine()
        self.pdf_parser = PDFMetadataExtractor()
        self.email_extractor = EmailIntelligenceExtractor()
        self.github_scanner = GitHubScanner()
        self.stackoverflow_scanner = StackOverflowScanner()
        self.analyzer = VerificationAnalyzer()
        
        # Initialize ML modules
        self.ml_pipeline = MLPipeline()
        self.dataset_manager = DatasetManager()
        self.nlp_analyzer = NLPAnalyzer()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dorking_queries": [],
            "extracted_metadata": [],
            "community_issues": [],
            "verification_gaps": [],
            "recommendations": [],
            "ml_insights": {},
            "nlp_analysis": {},
        }
    
    def run_dorking_search(self, chip_models: list, doc_types: list = None, output_file: str = None):
        """
        Run automated dorking searches.
        
        Args:
            chip_models: List of chip models to search for
            doc_types: Types of documents to search for
            output_file: Optional output file for queries
        """
        print("\n" + "=" * 70)
        print("PHASE 1: AUTOMATED DORKING - SEARCHING FOR TECHNICAL DOCUMENTATION")
        print("=" * 70)
        
        queries = self.dorking_engine.batch_generate_queries(
            chip_models=chip_models,
            doc_types=doc_types or ["datasheet", "errata", "reference_manual"],
            sites=["st.com", "mouser.com", "digikey.com"]
        )
        
        print(f"\n✓ Generated {len(self.dorking_engine.queries_generated)} search queries")
        print(f"  Chips analyzed: {len(queries)}")
        
        for chip, query_list in queries.items():
            print(f"\n  {chip}: {len(query_list)} queries")
            for i, query in enumerate(query_list[:2]):  # Show first 2
                print(f"    [{i+1}] {query}")
            if len(query_list) > 2:
                print(f"    ... and {len(query_list) - 2} more")
        
        self.results["dorking_queries"] = self.dorking_engine.queries_generated
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(self.dorking_engine.export_queries(format="txt"))
            print(f"\n✓ Queries exported to {output_file}")
    
    def run_pdf_parsing(self, pdf_directory: str = None):
        """
        Run PDF metadata extraction and parsing.
        
        Args:
            pdf_directory: Directory containing PDFs to analyze
        """
        print("\n" + "=" * 70)
        print("PHASE 2: METADATA PARSER - EXTRACTING INTELLIGENCE FROM PDFs")
        print("=" * 70)
        
        if pdf_directory and Path(pdf_directory).exists():
            metadata_results = self.pdf_parser.batch_extract(pdf_directory)
            self.results["extracted_metadata"] = metadata_results
            
            print(f"\n✓ Processed {len(metadata_results)} PDF files")
            
            # Generate CSV report
            csv_file = Path(pdf_directory).parent / "extracted_meta.csv"
            self.pdf_parser.generate_csv_report(str(csv_file))
        else:
            print("\n⚠ No PDF directory specified. Demonstrating with sample datasheet text...")
            
            # Demonstrate with sample text
            sample_text = """
            STM32F407VG Ultra-Performance Microcontroller
            Datasheet - Revision 9.0
            Date: 15-June-2023
            
            Contact: support@st.com
            Technical Support: technical-support@st.com
            Errata Updates: errata@st.com
            
            Part Number: STM32F407VGTx
            Package: LQFP144
            Pin Count: 144 pins
            Operating Frequency: 168 MHz
            """
            
            emails = self.pdf_parser.extract_emails(sample_text)
            versions = self.pdf_parser.extract_versions(sample_text)
            parts = self.pdf_parser.extract_part_numbers(sample_text)
            specs = self.pdf_parser.extract_technical_specs(sample_text)
            
            print(f"\n✓ Sample Extraction Results:")
            print(f"  Emails Found: {emails}")
            print(f"  Versions: {versions}")
            print(f"  Part Numbers: {parts}")
            print(f"  Technical Specs: {specs}")
    
    def run_community_analysis(self, chip_models: list):
        """
        Run community intelligence analysis.
        
        Args:
            chip_models: List of chip models to analyze
        """
        print("\n" + "=" * 70)
        print("PHASE 3: COMMUNITY PULSE - ANALYZING REAL-WORLD ISSUES")
        print("=" * 70)
        
        all_issues = []
        
        for chip in chip_models:
            print(f"\n  Analyzing {chip}...")
            
            # GitHub Analysis
            github_issues = self.github_scanner.search_repositories(chip, max_results=10)
            all_issues.extend(github_issues)
            print(f"    ✓ Found {len(github_issues)} issues on GitHub")
            
            # Stack Overflow Analysis
            so_questions = self.stackoverflow_scanner.search_questions(chip)
            print(f"    ✓ Found {len(so_questions)} Stack Overflow questions")
        
        self.results["community_issues"] = [
            {
                "title": issue.title,
                "severity": issue.severity,
                "type": issue.issue_type.value,
                "votes": issue.votes,
                "url": issue.url,
            }
            for issue in all_issues
        ]
        
        # Verification Analysis
        print(f"\n✓ Analyzing verification gaps from {len(all_issues)} issues...")
        gaps = self.analyzer.generate_verification_gaps(all_issues)
        recommendations = self.analyzer.create_test_plan_recommendations(all_issues)
        
        self.results["verification_gaps"] = gaps
        self.results["recommendations"] = recommendations
        
        print(f"\n  Verification Gaps Identified:")
        for gap_type, gap_list in gaps.items():
            if gap_list:
                print(f"    • {gap_type.replace('_', ' ').title()}: {len(gap_list)} issues")
        
        print(f"\n  Recommendations:")
        for rec in recommendations:
            print(f"    • {rec}")
    
    def generate_report(self, output_file: str = "semiintel_report.json"):
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Output file path for JSON report
        """
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE INTELLIGENCE REPORT")
        print("=" * 70)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n✓ Report generated: {output_file}")
        print(f"\nReport Summary:")
        print(f"  • Timestamp: {self.results['timestamp']}")
        print(f"  • Dorking Queries Generated: {len(self.results['dorking_queries'])}")
        print(f"  • PDF Metadata Extracted: {len(self.results['extracted_metadata'])}")
        print(f"  • Community Issues Found: {len(self.results['community_issues'])}")
        print(f"  • Verification Gaps: {sum(len(v) for v in self.results['verification_gaps'].values())}")
        print(f"  • Recommendations: {len(self.results['recommendations'])}")
    
    def run_ml_analysis(self, chip_models: list):
        """
        Run machine learning analysis on issues.
        
        Args:
            chip_models: List of chip models to analyze
        """
        print("\n" + "=" * 70)
        print("PHASE 4: MACHINE LEARNING INTELLIGENCE ANALYSIS")
        print("=" * 70)
        
        # Generate or collect issues
        test_issues = [
            f"{chip} UART transmission drops characters at high baud rates"
            for chip in chip_models
        ] + [
            f"{chip} I2C clock stretching causes system hang"
            for chip in chip_models
        ] + [
            f"{chip} DMA memory corruption in certain conditions"
            for chip in chip_models
        ]
        
        print(f"\n  Training ML models on {len(test_issues)} issues...")
        
        # Train ML pipeline
        training_results = self.ml_pipeline.train_all_models(test_issues)
        self.results["ml_insights"]["training_results"] = training_results
        
        print(f"\n  ✓ Severity Classifier trained: {training_results['severity_classifier']['mean_cv_score']:.2%} CV accuracy")
        print(f"  ✓ Issue Clusterer trained: Silhouette score {training_results['issue_clusterer']['silhouette_score']:.4f}")
        print(f"  ✓ Performance Predictor trained: {training_results['performance_predictor']['training_accuracy']:.2%} accuracy")
        
        # Make predictions on sample issues
        print(f"\n  Making predictions...")
        predictions = {}
        
        for issue in test_issues[:5]:
            pred = self.ml_pipeline.predict_all(issue)
            predictions[issue[:50]] = {
                "severity": pred["severity"].predicted_value,
                "severity_confidence": pred["severity"].confidence,
                "performance": pred["performance"].predicted_value,
            }
        
        self.results["ml_insights"]["predictions"] = predictions
        
        # Clustering analysis
        cluster_summary = self.ml_pipeline.issue_clusterer.get_cluster_summary(test_issues)
        self.results["ml_insights"]["clustering"] = {
            str(k): {
                "size": v["size"],
                "keywords": v["keywords"]
            }
            for k, v in cluster_summary.items()
            if v["size"] > 0
        }
        
        print(f"\n  ✓ Identified {len([c for c in cluster_summary.values() if c['size'] > 0])} issue clusters")
        for cluster_id, info in cluster_summary.items():
            if info["size"] > 0:
                print(f"    Cluster {cluster_id}: {info['size']} issues - {', '.join(info['keywords'])}")
    
    def run_nlp_analysis(self, chip_models: list):
        """
        Run NLP analysis on technical documentation.
        
        Args:
            chip_models: List of chip models to analyze
        """
        print("\n" + "=" * 70)
        print("PHASE 5: NATURAL LANGUAGE PROCESSING ANALYSIS")
        print("=" * 70)
        
        # Sample technical document
        tech_doc = f"""
        {chip_models[0]} Ultra-Performance Microcontroller Datasheet
        
        The {chip_models[0]} operates at frequencies up to 168 MHz with 1024KB flash memory
        and 192KB SRAM. Package type is LQFP144 with 144 pins. Operating voltage ranges
        from 2.0V to 3.6V. The device features UART, SPI, I2C, ADC, DMA, and USB interfaces.
        
        Critical Issue: UART transmission may drop characters at baud rates above 115200.
        Temperature range: -40°C to 85°C. Email support: technical-support@st.com
        """
        
        print(f"\n  Analyzing technical document...")
        
        # Named Entity Recognition
        ner = NamedEntityRecognizer()
        entities = ner.extract_entities(tech_doc)
        
        self.results["nlp_analysis"]["entities"] = {
            entity.entity_type: [e.text for e in entities if e.entity_type == entity.entity_type]
            for entity in entities
        }
        
        print(f"\n  ✓ Named Entity Recognition:")
        for entity_type, values in self.results["nlp_analysis"]["entities"].items():
            if values:
                print(f"    {entity_type}: {', '.join(set(values))}")
        
        # Complete NLP analysis
        analysis = self.nlp_analyzer.analyze_document(tech_doc)
        
        # Store results
        self.results["nlp_analysis"]["keywords"] = [
            {"term": kw, "score": float(score)} for kw, score in analysis["keywords"]
        ]
        self.results["nlp_analysis"]["technical_terms"] = analysis["technical_terms"]
        self.results["nlp_analysis"]["sentiment"] = {
            "sentiment": analysis["sentiment"][0],
            "confidence": float(analysis["sentiment"][1])
        }
        
        print(f"\n  ✓ Keyword Extraction: {len(analysis['keywords'])} top keywords extracted")
        print(f"  ✓ Technical Terms: {', '.join(analysis['technical_terms'][:5])}")
        print(f"  ✓ Sentiment: {analysis['sentiment'][0]} ({analysis['sentiment'][1]:.2%} confidence)")
    
    def load_kaggle_datasets(self):
        """
        Load and demonstrate Kaggle datasets.
        """
        print("\n" + "=" * 70)
        print("KAGGLE DATASETS - AVAILABLE FOR ML TRAINING")
        print("=" * 70)
        
        datasets = KaggleDatasetRegistry.list_datasets()
        print(f"\n  Total datasets available: {len(datasets)}")
        print(f"  Total storage required: {KaggleDatasetRegistry.total_storage_required() / 1024:.1f} GB")
        
        print(f"\n  Loading sample datasets...")
        
        # Load synthetic datasets
        issues_df = self.dataset_manager.load_dataset("github_issues")
        specs_df = self.dataset_manager.load_dataset("microcontroller_specs")
        
        if issues_df is not None:
            print(f"  ✓ GitHub Issues: {len(issues_df)} records, {len(issues_df.columns)} features")
        
        if specs_df is not None:
            print(f"  ✓ Microcontroller Specs: {len(specs_df)} records, {len(specs_df.columns)} features")
        
        return {
            "github_issues": issues_df,
            "microcontroller_specs": specs_df,
        }


def main():
    """Main entry point for the SEMIINTEL tool."""
    parser = argparse.ArgumentParser(
        description="SEMIINTEL: Semiconductor Intelligence Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dorking STM32F407VG STM32H7 --output queries.txt
  python main.py --community STM32F407VG --report analysis.json
  python main.py --all STM32F407VG STM32H7 --output-dir ./results
  python main.py --pdf ./datasheets
        """
    )
    
    parser.add_argument(
        "--dorking",
        nargs="+",
        help="Run dorking module for specified chip models"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Run PDF parser on specified directory"
    )
    parser.add_argument(
        "--community",
        nargs="+",
        help="Run community intelligence analysis for specified chip models"
    )
    parser.add_argument(
        "--all",
        nargs="+",
        help="Run all analysis modules for specified chip models"
    )
    parser.add_argument(
        "--list-chips",
        action="store_true",
        help="List available STMicroelectronics chip models"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for dorking queries"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="semiintel_report.json",
        help="Output file for comprehensive report (default: semiintel_report.json)"
    )
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=["datasheet", "errata"],
        help="Document types to search for (default: datasheet errata)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--ml",
        nargs="+",
        help="Run machine learning analysis for specified chip models"
    )
    parser.add_argument(
        "--nlp",
        nargs="+",
        help="Run NLP analysis for specified chip models"
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Display and load available Kaggle datasets"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 70)
    print("SEMIINTEL - SEMICONDUCTOR INTELLIGENCE AUTOMATION TOOL")
    print("For IC Digital Design & Verification")
    print("=" * 70)
    print("Project: STMicroelectronics Portfolio Analysis")
    print("=" * 70)
    
    # Handle chip list
    if args.list_chips:
        print("\nAvailable STMicroelectronics Chip Models:")
        for i, chip in enumerate(STM_CHIP_MODELS, 1):
            print(f"  {i:2d}. {chip}")
        return
    
    # Initialize CLI
    cli = SemiIntelCLI()
    
    # Execute requested operations
    if args.dorking:
        cli.run_dorking_search(args.dorking, args.doc_types, args.output)
    
    if args.pdf:
        cli.run_pdf_parsing(args.pdf)
    
    if args.community:
        cli.run_community_analysis(args.community)
    
    if args.ml:
        cli.run_ml_analysis(args.ml)
    
    if args.nlp:
        cli.run_nlp_analysis(args.nlp)
    
    if args.datasets:
        datasets = cli.load_kaggle_datasets()
    
    if args.all:
        cli.run_dorking_search(args.all, args.doc_types, args.output)
        cli.run_pdf_parsing()
        cli.run_community_analysis(args.all)
        cli.run_ml_analysis(args.all)
        cli.run_nlp_analysis(args.all)
        cli.load_kaggle_datasets()
    
    # Generate report if any analysis was run
    if args.dorking or args.pdf or args.community or args.all or args.ml or args.nlp or args.datasets:
        cli.generate_report(args.report)
    
    if args.verbose:
        print(f"\n[DEBUG] Results stored: {list(cli.results.keys())}")
    
    print("\n" + "=" * 70)
    print("✓ SEMIINTEL ANALYSIS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
