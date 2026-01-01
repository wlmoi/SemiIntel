"""
Dorking Engine Module
Implements automated Google Dorking logic to search for semiconductor datasheets,
errata sheets, and technical documentation across STMicroelectronics and distributor sites.

This module demonstrates OSINT techniques repurposed for Technical Intelligence gathering.
"""

import re
from typing import List, Dict, Tuple
from urllib.parse import urlencode


class DorkingEngine:
    """
    Automated Dorking Engine for discovering technical documentation.
    
    Implements Google Dorking queries using filetype and site operators to locate:
    - Datasheets (filetype:pdf)
    - Reference manuals
    - Errata sheets
    - Technical application notes
    """
    
    # Default search domains for semiconductor documentation
    DEFAULT_SITES = [
        "st.com",  # Official STMicroelectronics site
        "distributor.com",  # Generic distributor
        "mouser.com",
        "digikey.com",
        "wyle.com"
    ]
    
    # Document types and keywords for targeted searches
    DOCUMENT_TYPES = {
        "datasheet": {
            "keywords": ["datasheet", "DS"],
            "filetype": "pdf"
        },
        "reference_manual": {
            "keywords": ["reference", "manual", "RM"],
            "filetype": "pdf"
        },
        "errata": {
            "keywords": ["errata", "ES", "revision", "history"],
            "filetype": "pdf"
        },
        "application_note": {
            "keywords": ["application note", "AN", "reference", "guide"],
            "filetype": "pdf"
        },
        "programming_manual": {
            "keywords": ["programming", "PM", "manual"],
            "filetype": "pdf"
        }
    }
    
    def __init__(self, sites: List[str] = None):
        """
        Initialize the Dorking Engine.
        
        Args:
            sites: List of domains to search. Defaults to DEFAULT_SITES.
        """
        self.sites = sites or self.DEFAULT_SITES
        self.queries_generated = []
    
    def generate_dork_query(self, 
                           chip_model: str, 
                           doc_type: str = "datasheet",
                           site: str = None,
                           revision: str = None) -> str:
        """
        Generate a Google Dorking query string.
        
        Args:
            chip_model: The semiconductor chip model (e.g., "STM32F4", "STM32H7")
            doc_type: Type of document to search for (datasheet, errata, etc.)
            site: Specific domain to search. If None, uses all DEFAULT_SITES
            revision: Optional revision number to filter results
            
        Returns:
            A formatted Google Dorking query string
        """
        if doc_type not in self.DOCUMENT_TYPES:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        doc_config = self.DOCUMENT_TYPES[doc_type]
        keywords = " OR ".join(doc_config["keywords"])
        filetype = doc_config["filetype"]
        
        # Build the base query
        query_parts = [
            f'"{chip_model}"',
            f'({keywords})',
            f'filetype:{filetype}'
        ]
        
        # Add site restriction if specified
        if site:
            query_parts.append(f'site:{site}')
        
        # Add revision filter if specified
        if revision:
            query_parts.append(f'"{revision}"')
        
        query = " ".join(query_parts)
        return query
    
    def batch_generate_queries(self, 
                              chip_models: List[str],
                              doc_types: List[str] = None,
                              sites: List[str] = None) -> Dict[str, List[str]]:
        """
        Generate multiple dorking queries for batch processing.
        
        Args:
            chip_models: List of chip models to search for
            doc_types: List of document types. Defaults to all types.
            sites: List of sites to search. Defaults to DEFAULT_SITES.
            
        Returns:
            Dictionary mapping chip models to lists of dorking queries
        """
        if doc_types is None:
            doc_types = list(self.DOCUMENT_TYPES.keys())
        
        if sites is None:
            sites = self.sites
        
        batch_queries = {}
        
        for chip_model in chip_models:
            batch_queries[chip_model] = []
            
            for doc_type in doc_types:
                for site in sites:
                    query = self.generate_dork_query(chip_model, doc_type, site)
                    batch_queries[chip_model].append(query)
                    self.queries_generated.append(query)
        
        return batch_queries
    
    def parse_query_results(self, results_text: str) -> List[Dict[str, str]]:
        """
        Parse search results to extract document URLs and metadata.
        
        Args:
            results_text: Raw text from search results
            
        Returns:
            List of dictionaries containing extracted metadata
        """
        extracted_data = []
        
        # Regex patterns for extracting URLs and metadata
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        date_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
        
        urls = re.findall(url_pattern, results_text)
        
        for url in urls:
            entry = {
                "url": url,
                "domain": self._extract_domain(url),
                "filetype": self._extract_filetype(url)
            }
            extracted_data.append(entry)
        
        return extracted_data
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else ""
    
    def _extract_filetype(self, url: str) -> str:
        """Extract file type from URL."""
        match = re.search(r'\.([a-z]+)$', url, re.IGNORECASE)
        return match.group(1).lower() if match else "unknown"
    
    def format_for_google(self, query: str) -> str:
        """
        Format query string for actual Google Search URL.
        
        Args:
            query: The dorking query string
            
        Returns:
            A complete Google Search URL with parameters
        """
        base_url = "https://www.google.com/search"
        params = {"q": query}
        return f"{base_url}?{urlencode(params)}"
    
    def export_queries(self, format: str = "txt") -> str:
        """
        Export generated queries in specified format.
        
        Args:
            format: Export format ('txt', 'csv', or 'json')
            
        Returns:
            Formatted string of queries
        """
        if format == "txt":
            return "\n".join(self.queries_generated)
        elif format == "csv":
            return "\n".join([f'"{q}"' for q in self.queries_generated])
        elif format == "json":
            import json
            return json.dumps({"queries": self.queries_generated}, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Common STMicroelectronics chip models for targeted searches
STM_CHIP_MODELS = [
    "STM32F4", "STM32F1", "STM32F0", "STM32F7",  # ARM Cortex-M series
    "STM32H7", "STM32L0", "STM32L1", "STM32L4",  # Low-power variants
    "STM32G0", "STM32G4",                        # General-purpose
    "STM32WB", "STM32MP1",                       # Wireless, Multicore
    "STM8",                                       # 8-bit series
]


def main():
    """Example usage of the Dorking Engine."""
    engine = DorkingEngine()
    
    # Example 1: Single query for STM32F4 datasheet
    print("=" * 60)
    print("EXAMPLE 1: Single Dorking Query")
    print("=" * 60)
    query = engine.generate_dork_query("STM32F407VG", "datasheet", "st.com")
    print(f"Query: {query}\n")
    print(f"Google URL: {engine.format_for_google(query)}\n")
    
    # Example 2: Batch queries for multiple chips
    print("=" * 60)
    print("EXAMPLE 2: Batch Query Generation")
    print("=" * 60)
    batch = engine.batch_generate_queries(
        chip_models=["STM32F4", "STM32H7"],
        doc_types=["datasheet", "errata"],
        sites=["st.com", "mouser.com"]
    )
    
    for chip, queries in batch.items():
        print(f"\n{chip}:")
        for q in queries:
            print(f"  - {q}")
    
    # Example 3: Export queries
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Export Generated Queries")
    print("=" * 60)
    print(f"Total queries generated: {len(engine.queries_generated)}")
    print("\nFirst 3 queries:")
    for q in engine.queries_generated[:3]:
        print(f"  {q}")


if __name__ == "__main__":
    main()
