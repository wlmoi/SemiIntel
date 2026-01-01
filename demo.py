#!/usr/bin/env python3
"""
SEMIINTEL ML Demo - Complete Machine Learning Pipeline
Demonstrates all ML/NLP capabilities with sample data.

Usage:
    python demo.py --full              # Run complete pipeline
    python demo.py --ml                # Run only ML analysis
    python demo.py --nlp               # Run only NLP analysis
    python demo.py --datasets          # Show dataset registry
"""

import sys
import argparse
from modules.ml_analyzer import MLPipeline, SeverityClassifier, IssueClusterer
from modules.dataset_loader import DatasetManager, KaggleDatasetRegistry, SyntheticDataGenerator
from modules.nlp_analyzer import NLPAnalyzer, NamedEntityRecognizer


def demo_ml_pipeline():
    """Demonstrate complete ML pipeline."""
    print("\n" + "=" * 80)
    print("SEMIINTEL ML PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Training data
    training_issues = [
        "STM32F407VG UART transmission drops characters at 115200 baud",
        "I2C clock stretching causes system deadlock and requires hard reset",
        "DMA memory corruption in certain access patterns at high frequency",
        "USB enumeration fails intermittently under heavy load",
        "ADC sampling produces incorrect values in power-saving modes",
        "SPI communication errors at speeds above 10 MHz",
        "Timer overflow interrupt not triggered on edge cases",
        "GPIO interrupt latency causes race conditions",
        "Missing documentation for USB device emulation",
        "Datasheet section 3.2 has conflicting timing specs",
    ] * 100
    
    # Train models
    print("\n1. TRAINING MODELS")
    print("-" * 80)
    training_metrics = pipeline.train_all_models(training_issues)
    
    for model_name, metrics in training_metrics.items():
        print(f"\n✓ {model_name.replace('_', ' ').title()}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Make predictions
    print("\n\n2. MAKING PREDICTIONS")
    print("-" * 80)
    
    test_issues = [
        "I2C deadlock freezes system - critical issue",
        "Would be nice to have more USB examples",
        "Timer interrupt handling bug causes data loss",
    ]
    
    for issue in test_issues:
        print(f"\n📝 Issue: {issue}")
        predictions = pipeline.predict_all(issue)
        
        for pred_type, pred in predictions.items():
            print(f"   {pred_type}: {pred.predicted_value} "
                  f"(confidence: {pred.confidence:.1%})")
    
    # Clustering analysis
    print("\n\n3. ISSUE CLUSTERING")
    print("-" * 80)
    
    sample_cluster_texts = training_issues[:15]
    cluster_summary = pipeline.issue_clusterer.get_cluster_summary(sample_cluster_texts)
    
    print("\nCluster Breakdown:")
    for cluster_id, info in cluster_summary.items():
        if info["size"] > 0:
            print(f"\n  Cluster {cluster_id} ({info['size']} issues):")
            print(f"    Topics: {', '.join(info['keywords'])}")
            if info['sample_issues']:
                print(f"    Example: {info['sample_issues'][0][:70]}...")
    
    print("\n" + "=" * 80 + "\n")


def demo_nlp_analysis():
    """Demonstrate NLP capabilities."""
    print("\n" + "=" * 80)
    print("SEMIINTEL NLP ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    analyzer = NLPAnalyzer()
    
    # Sample technical document
    document = """
    STM32F407VG Ultra-Performance ARM Cortex-M4 Microcontroller
    
    Part Number: STM32F407VGTx
    Package: LQFP144 (144-pin)
    Operating Frequency: 168 MHz maximum
    Flash Memory: 1024 KB
    SRAM: 192 KB
    Operating Voltage: 2.0V to 3.6V
    Temperature Range: -40°C to 85°C
    
    Peripherals:
    - 3x UART interfaces
    - 3x SPI/I2S interfaces  
    - 3x I2C interfaces
    - 16-channel 12-bit ADC
    - 2-channel 12-bit DAC
    - 12x 16-bit timers
    - 2x 32-bit timers
    - 2x watchdog timers
    - 12-channel DMA controller
    - Full-speed USB 2.0 device/host
    
    Known Issue: UART transmission may drop characters at baud rates above 115200
    Support: technical-support@st.com or support@st.com
    Revision: Rev Z (2023-06-15)
    
    Critical Timing Specification (pg 45):
    I2C Clock Stretching: Must complete within 100µs
    DMA Transfer: Minimum 2 cycles between requests
    """
    
    print("\n1. NAMED ENTITY RECOGNITION")
    print("-" * 80)
    
    entities = analyzer.ner.extract_entities(document)
    entity_types = {}
    
    for entity in entities:
        if entity.entity_type not in entity_types:
            entity_types[entity.entity_type] = []
        entity_types[entity.entity_type].append(entity.text)
    
    for etype, values in entity_types.items():
        unique_values = list(set(values))
        print(f"\n{etype.upper()}: {', '.join(unique_values[:5])}")
    
    print("\n\n2. KEYWORD EXTRACTION")
    print("-" * 80)
    
    keywords = analyzer.keyword_extractor.extract_keywords(document, top_k=10)
    print("\nTop Keywords by TF-IDF Score:")
    for keyword, score in keywords:
        print(f"  {keyword:25} {score:6.4f}")
    
    print("\n\n3. TECHNICAL TERM EXTRACTION")
    print("-" * 80)
    
    tech_terms = analyzer.keyword_extractor.extract_technical_terms(document)
    print(f"\nTechnical Terms Found: {', '.join(tech_terms)}")
    
    print("\n\n4. SENTIMENT ANALYSIS")
    print("-" * 80)
    
    sentiments_to_analyze = [
        "This chip is absolutely fantastic and works perfectly!",
        "The datasheet is confusing and the chip doesn't work correctly.",
        "The documentation could be improved but it's functional.",
    ]
    
    for text in sentiments_to_analyze:
        sentiment, confidence = analyzer.sentiment_analyzer.analyze_sentiment(text)
        print(f"\n\"{text}\"")
        print(f"  Sentiment: {sentiment} ({confidence:.1%} confidence)")
    
    print("\n" + "=" * 80 + "\n")


def demo_datasets():
    """Demonstrate Kaggle dataset registry."""
    print("\n" + "=" * 80)
    print("KAGGLE DATASET REGISTRY - SEMIINTEL")
    print("=" * 80)
    
    datasets = KaggleDatasetRegistry.list_datasets()
    total_size = KaggleDatasetRegistry.total_storage_required()
    
    print(f"\n📊 Total Datasets: {len(datasets)}")
    print(f"📦 Total Storage Required: {total_size/1024:.1f} GB\n")
    
    print("Available Datasets:")
    print("-" * 80)
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   📝 Description: {ds['description']}")
        print(f"   📊 Size: {ds['size_mb']:,} MB | Records: {ds['rows']:,} | Features: {ds['columns']}")
        print(f"   🎯 Use: {ds['primary_use']}")
        print(f"   📅 Updated: {ds['last_updated']}")
    
    # Load sample datasets
    print("\n\n" + "=" * 80)
    print("LOADING SAMPLE DATASETS")
    print("=" * 80)
    
    manager = DatasetManager()
    
    print("\n1. GitHub Issues Dataset")
    print("-" * 80)
    issues_df = manager.load_dataset("github_issues", use_synthetic=True)
    if issues_df is not None:
        print(f"Shape: {issues_df.shape[0]} rows × {issues_df.shape[1]} columns")
        print(f"\nSample Data:")
        print(issues_df[['title', 'severity', 'votes']].head(3).to_string(index=False))
    
    print("\n\n2. Microcontroller Specifications Dataset")
    print("-" * 80)
    specs_df = manager.load_dataset("microcontroller_specs", use_synthetic=True)
    if specs_df is not None:
        print(f"Shape: {specs_df.shape[0]} rows × {specs_df.shape[1]} columns")
        print(f"\nSample Data:")
        print(specs_df[['part_number', 'frequency_mhz', 'flash_kb', 'price_usd']].head(3).to_string(index=False))
    
    print("\n\n3. Bug Report Dataset")
    print("-" * 80)
    bugs_df = manager.load_dataset("community_bugs", use_synthetic=True)
    if bugs_df is not None:
        print(f"Shape: {bugs_df.shape[0]} rows × {bugs_df.shape[1]} columns")
        print(f"\nSample Data:")
        print(bugs_df[['component', 'severity', 'status']].value_counts().head(5).to_string())
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="SEMIINTEL ML/NLP Pipeline Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --full       # Run complete ML + NLP pipeline
  python demo.py --ml         # ML analysis only
  python demo.py --nlp        # NLP analysis only
  python demo.py --datasets   # Show Kaggle datasets
        """
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete ML and NLP pipeline"
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Run ML pipeline demonstration"
    )
    parser.add_argument(
        "--nlp",
        action="store_true",
        help="Run NLP analysis demonstration"
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Show Kaggle datasets and load samples"
    )
    
    args = parser.parse_args()
    
    if not any([args.full, args.ml, args.nlp, args.datasets]):
        parser.print_help()
        return
    
    if args.full:
        demo_ml_pipeline()
        demo_nlp_analysis()
        demo_datasets()
    else:
        if args.ml:
            demo_ml_pipeline()
        if args.nlp:
            demo_nlp_analysis()
        if args.datasets:
            demo_datasets()
    
    print("\n✓ Demo completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
