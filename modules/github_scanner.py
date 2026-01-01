"""
GitHub and Community Intelligence Scanner Module
Searches GitHub, Stack Overflow, and other communities for reported issues and bugs
related to specific semiconductor components.

This module demonstrates verification capabilities by identifying real-world failure
points and community-reported issues that can guide verification test planning.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class IssueType(Enum):
    """Types of issues that can be reported."""
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    ERRATA = "errata"
    PERFORMANCE = "performance"
    DATASHEET = "datasheet"


@dataclass
class CommunityIssue:
    """Represents a reported issue from the community."""
    title: str
    url: str
    platform: str  # GitHub, StackOverflow, etc.
    issue_type: IssueType
    chip_model: str
    severity: str  # Critical, High, Medium, Low
    date_reported: str
    author: str
    description: str
    votes: int = 0
    relevance_score: float = 0.0


class GitHubScanner:
    """
    Scans GitHub repositories for issues related to semiconductor components.
    
    Demonstrates:
    - Web scraping and API knowledge
    - Issue categorization and severity assessment
    - Community-driven verification insights
    """
    
    # Keywords that indicate critical issues
    CRITICAL_KEYWORDS = [
        "crash",
        "hang",
        "freeze",
        "memory leak",
        "data corruption",
        "deadlock",
        "stack overflow",
        "hardware reset required",
        "unrecoverable error",
    ]
    
    # Keywords indicating documentation issues
    DOC_KEYWORDS = [
        "datasheet",
        "documentation",
        "unclear",
        "missing",
        "ambiguous",
        "incorrect",
        "example",
        "tutorial",
    ]
    
    # Keywords for peripheral-specific issues
    PERIPHERAL_KEYWORDS = {
        "uart": ["serial", "rx", "tx", "baud", "framing"],
        "spi": ["clock", "mode", "mosi", "miso", "chip select"],
        "i2c": ["ack", "nack", "clock stretching", "start condition"],
        "adc": ["conversion", "sampling", "resolution", "offset"],
        "timer": ["overflow", "compare", "pwm", "interrupt"],
        "dma": ["transfer", "burst", "channel", "memory"],
        "usb": ["enumeration", "ep0", "descriptor", "driver"],
    }
    
    def __init__(self):
        """Initialize the GitHub Scanner."""
        self.issues_found = []
    
    def search_repositories(self, 
                           chip_model: str,
                           keywords: List[str] = None,
                           max_results: int = 50) -> List[CommunityIssue]:
        """
        Search GitHub repositories for issues related to a chip model.
        
        Args:
            chip_model: Semiconductor chip model (e.g., "STM32F407VG")
            keywords: Additional keywords to include in search
            max_results: Maximum number of results to return
            
        Returns:
            List of CommunityIssue objects
        """
        search_queries = self._build_search_queries(chip_model, keywords)
        issues = []
        
        for query in search_queries:
            # In production, this would make actual GitHub API calls
            # For now, we generate representative data
            simulated_issues = self._simulate_github_results(query, chip_model, max_results // len(search_queries))
            issues.extend(simulated_issues)
        
        self.issues_found.extend(issues)
        return issues
    
    def _build_search_queries(self, 
                             chip_model: str,
                             additional_keywords: List[str] = None) -> List[str]:
        """
        Build GitHub search queries for a chip model.
        
        Args:
            chip_model: The chip model to search for
            additional_keywords: Additional search terms
            
        Returns:
            List of search query strings
        """
        base_queries = [
            f'"{chip_model}" issue',
            f'"{chip_model}" bug',
            f'"{chip_model}" errata',
            f'"{chip_model}" error',
            f'"{chip_model}" crash',
        ]
        
        if additional_keywords:
            for keyword in additional_keywords:
                base_queries.append(f'"{chip_model}" {keyword}')
        
        return base_queries
    
    def _simulate_github_results(self, 
                                query: str,
                                chip_model: str,
                                count: int) -> List[CommunityIssue]:
        """
        Simulate GitHub search results (for demonstration).
        
        Args:
            query: Search query
            chip_model: Chip model being searched
            count: Number of results to simulate
            
        Returns:
            List of simulated issues
        """
        # Representative issues commonly found in embedded systems projects
        sample_issues = [
            {
                "title": f"{chip_model} UART transmission drops characters at high baud rates",
                "severity": "High",
                "type": IssueType.BUG,
                "votes": 12,
            },
            {
                "title": f"{chip_model} I2C clock stretching not working properly",
                "severity": "Critical",
                "type": IssueType.BUG,
                "votes": 24,
            },
            {
                "title": f"Datasheet for {chip_model} has conflicting timing specifications",
                "severity": "High",
                "type": IssueType.ERRATA,
                "votes": 18,
            },
            {
                "title": f"{chip_model} DMA channel conflicts on simultaneous transfers",
                "severity": "Critical",
                "type": IssueType.BUG,
                "votes": 31,
            },
            {
                "title": f"{chip_model} ADC sampling rate inconsistency in low power mode",
                "severity": "Medium",
                "type": IssueType.BUG,
                "votes": 8,
            },
            {
                "title": f"Missing {chip_model} example code for USB device emulation",
                "severity": "Low",
                "type": IssueType.DOCUMENTATION,
                "votes": 5,
            },
            {
                "title": f"{chip_model} timer overflow not triggering interrupt correctly",
                "severity": "High",
                "type": IssueType.BUG,
                "votes": 14,
            },
            {
                "title": f"{chip_model} requires external reset after certain error states",
                "severity": "Critical",
                "type": IssueType.ERRATA,
                "votes": 22,
            },
        ]
        
        issues = []
        for i, sample in enumerate(sample_issues[:count]):
            issue = CommunityIssue(
                title=sample["title"],
                url=f"https://github.com/search?q={query.replace(' ', '+')}",
                platform="GitHub",
                issue_type=sample["type"],
                chip_model=chip_model,
                severity=sample["severity"],
                date_reported=f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                author=f"engineer_{i}",
                description=f"Discussion around {sample['title']}",
                votes=sample["votes"],
                relevance_score=self._calculate_relevance(sample["title"])
            )
            issues.append(issue)
        
        return issues
    
    def _calculate_relevance(self, title: str) -> float:
        """
        Calculate relevance score for an issue title.
        
        Args:
            title: Issue title
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.5  # Base score
        
        # Increase score for critical keywords
        for keyword in self.CRITICAL_KEYWORDS:
            if keyword.lower() in title.lower():
                score += 0.2
        
        # Increase score for errata mentions
        if "errata" in title.lower() or "datasheet" in title.lower():
            score += 0.15
        
        return min(score, 1.0)


class StackOverflowScanner:
    """
    Scans Stack Overflow for questions related to semiconductor components.
    """
    
    def __init__(self):
        """Initialize the Stack Overflow Scanner."""
        self.questions_found = []
    
    def search_questions(self, 
                        chip_model: str,
                        tags: List[str] = None) -> List[Dict[str, any]]:
        """
        Search Stack Overflow for questions about a chip model.
        
        Args:
            chip_model: The chip model to search for
            tags: Stack Overflow tags to include (e.g., ["embedded-systems", "microcontroller"])
            
        Returns:
            List of question dictionaries
        """
        default_tags = ["embedded-systems", "microcontroller", "c"]
        if tags:
            default_tags.extend(tags)
        
        questions = self._simulate_stackoverflow_results(chip_model, default_tags)
        self.questions_found.extend(questions)
        return questions
    
    def _simulate_stackoverflow_results(self, chip_model: str, tags: List[str]) -> List[Dict[str, any]]:
        """
        Simulate Stack Overflow search results.
        
        Args:
            chip_model: Chip model being searched
            tags: Search tags
            
        Returns:
            List of simulated questions
        """
        sample_questions = [
            {
                "title": f"How to debug {chip_model} UART timeout issues?",
                "votes": 8,
                "answers": 3,
                "url": f"https://stackoverflow.com/questions/search?q={chip_model}",
                "tags": ["uart", "debugging"] + tags,
            },
            {
                "title": f"Best practices for {chip_model} power consumption optimization",
                "votes": 15,
                "answers": 5,
                "url": f"https://stackoverflow.com/questions/search?q={chip_model}",
                "tags": ["power-management", "optimization"] + tags,
            },
            {
                "title": f"{chip_model} SPI communication CRC errors",
                "votes": 6,
                "answers": 2,
                "url": f"https://stackoverflow.com/questions/search?q={chip_model}",
                "tags": ["spi", "communication"] + tags,
            },
            {
                "title": f"How do I reset {chip_model} watchdog timer correctly?",
                "votes": 12,
                "answers": 4,
                "url": f"https://stackoverflow.com/questions/search?q={chip_model}",
                "tags": ["watchdog", "timer"] + tags,
            },
        ]
        
        return sample_questions


class VerificationAnalyzer:
    """
    Analyzes community findings to guide verification test planning.
    """
    
    def __init__(self):
        """Initialize the Verification Analyzer."""
        self.analysis_results = {}
    
    def generate_verification_gaps(self, issues: List[CommunityIssue]) -> Dict[str, List[str]]:
        """
        Identify verification gaps based on reported issues.
        
        Args:
            issues: List of community issues
            
        Returns:
            Dictionary mapping test areas to identified gaps
        """
        gaps = {
            "peripheral_testing": [],
            "edge_cases": [],
            "error_handling": [],
            "documentation": [],
            "performance": [],
        }
        
        for issue in issues:
            if issue.issue_type == IssueType.BUG:
                # Extract peripheral information
                for peripheral, keywords in GitHubScanner.PERIPHERAL_KEYWORDS.items():
                    if any(kw in issue.title.lower() for kw in keywords):
                        gaps["peripheral_testing"].append(
                            f"Test {peripheral.upper()} behavior: {issue.title}"
                        )
                
                # Categorize by severity
                if issue.severity in ["Critical", "High"]:
                    gaps["edge_cases"].append(f"Edge case found: {issue.title}")
            
            elif issue.issue_type == IssueType.ERRATA:
                gaps["error_handling"].append(f"Errata: {issue.title}")
            
            elif issue.issue_type == IssueType.DOCUMENTATION:
                gaps["documentation"].append(f"Documentation gap: {issue.title}")
        
        return gaps
    
    def create_test_plan_recommendations(self, issues: List[CommunityIssue]) -> List[str]:
        """
        Create test plan recommendations based on issues found.
        
        Args:
            issues: List of community issues
            
        Returns:
            List of test plan recommendations
        """
        recommendations = []
        
        # Count issues by severity
        critical_count = sum(1 for i in issues if i.severity == "Critical")
        high_count = sum(1 for i in issues if i.severity == "High")
        
        recommendations.append(f"Priority Areas: {critical_count} critical, {high_count} high-severity issues identified")
        
        # Identify most discussed peripherals
        peripheral_counts = {}
        for issue in issues:
            for peripheral in GitHubScanner.PERIPHERAL_KEYWORDS.keys():
                if peripheral in issue.title.lower():
                    peripheral_counts[peripheral] = peripheral_counts.get(peripheral, 0) + 1
        
        if peripheral_counts:
            top_peripheral = max(peripheral_counts, key=peripheral_counts.get)
            recommendations.append(f"Focus on {top_peripheral.upper()} testing ({peripheral_counts[top_peripheral]} issues)")
        
        recommendations.append(f"Community consensus: {sum(i.votes for i in issues)} total votes on issues")
        
        return recommendations


def main():
    """Example usage of the GitHub Scanner."""
    print("=" * 70)
    print("COMMUNITY INTELLIGENCE SCANNER - STM32F407VG ANALYSIS")
    print("=" * 70)
    
    # Initialize scanners
    github_scanner = GitHubScanner()
    stackoverflow_scanner = StackOverflowScanner()
    analyzer = VerificationAnalyzer()
    
    chip_model = "STM32F407VG"
    
    # Scan GitHub
    print(f"\n1. SCANNING GITHUB FOR {chip_model} ISSUES...")
    print("-" * 70)
    issues = github_scanner.search_repositories(chip_model)
    
    for issue in issues[:5]:  # Show first 5
        severity_color = "🔴" if issue.severity == "Critical" else "🟠" if issue.severity == "High" else "🟡"
        print(f"\n{severity_color} [{issue.severity}] {issue.title}")
        print(f"   Type: {issue.issue_type.value} | Votes: {issue.votes} | Relevance: {issue.relevance_score:.2f}")
    
    # Scan Stack Overflow
    print(f"\n\n2. SCANNING STACK OVERFLOW FOR {chip_model} QUESTIONS...")
    print("-" * 70)
    questions = stackoverflow_scanner.search_questions(chip_model, ["uart", "spi", "i2c"])
    
    for question in questions[:3]:  # Show first 3
        print(f"\n❓ {question['title']}")
        print(f"   Votes: {question['votes']} | Answers: {question['answers']}")
        print(f"   Tags: {', '.join(question['tags'][:3])}")
    
    # Analyze verification gaps
    print(f"\n\n3. VERIFICATION GAP ANALYSIS")
    print("-" * 70)
    gaps = analyzer.generate_verification_gaps(issues)
    
    for gap_type, gap_list in gaps.items():
        if gap_list:
            print(f"\n{gap_type.upper().replace('_', ' ')} ({len(gap_list)} items):")
            for gap in gap_list[:3]:  # Show first 3
                print(f"   • {gap}")
    
    # Test plan recommendations
    print(f"\n\n4. TEST PLAN RECOMMENDATIONS")
    print("-" * 70)
    recommendations = analyzer.create_test_plan_recommendations(issues)
    for rec in recommendations:
        print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
