"""
PDF Parser Module
Extracts metadata and contact information from technical documentation PDFs.

This module demonstrates data extraction and text parsing capabilities essential
for IC Design verification, particularly handling complex technical documents.
"""

import re
import os
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime


class PDFMetadataExtractor:
    """
    Extracts metadata and intelligence from PDF files.
    
    Capabilities:
    - Parse PDF metadata (Author, Creation Date, Version)
    - Extract email addresses using regex patterns
    - Identify revision information
    - Extract contact information and support channels
    """
    
    # Regex patterns for common email formats
    EMAIL_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Standard email
        r'support@[a-z0-9.-]+\.[a-z]{2,}',  # Support addresses
        r'technical@[a-z0-9.-]+\.[a-z]{2,}',  # Technical support
        r'errata@[a-z0-9.-]+\.[a-z]{2,}',  # Errata updates
    ]
    
    # Patterns for extracting revision and version info
    VERSION_PATTERNS = [
        r'[Rr]evision\s*:?\s*([A-Za-z0-9._-]+)',
        r'[Vv]ersion\s*:?\s*([A-Za-z0-9._-]+)',
        r'Rev\.?\s*([A-Za-z0-9._-]+)',
        r'V([0-9]+\.[0-9]+)',
        r'([A-Z]\d{1,3})\s*[Rr]evision',
    ]
    
    # Patterns for technical metadata
    TECHNICAL_PATTERNS = {
        "part_number": r'\b(?:STM|LPC|NXP|ARM)\d+[A-Z0-9]+\b',
        "date": r'(\d{1,2}[/-]?\d{1,2}[/-]?\d{2,4})',
        "package_type": r'(BGA|LQFP|QFP|DIP|SOIC|SO8|SO16|TFBGA)\d*',
        "pin_count": r'(\d+)\s*[Pp]in',
        "frequency": r'(\d+)\s*[Mm]?[Hh]z',
    }
    
    def __init__(self):
        """Initialize the PDF Metadata Extractor."""
        self.extracted_data = []
    
    def extract_metadata(self, file_path: str) -> Dict[str, any]:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "file_size_kb": os.path.getsize(file_path) / 1024 if os.path.exists(file_path) else 0,
            "modified_date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat() if os.path.exists(file_path) else None,
            "emails": [],
            "revisions": [],
            "part_numbers": [],
            "technical_specs": {},
        }
        
        # Try to read PDF text (simplified - actual implementation would use PyPDF2)
        try:
            pdf_text = self._read_pdf_text(file_path)
            metadata["emails"] = self.extract_emails(pdf_text)
            metadata["revisions"] = self.extract_versions(pdf_text)
            metadata["part_numbers"] = self.extract_part_numbers(pdf_text)
            metadata["technical_specs"] = self.extract_technical_specs(pdf_text)
        except Exception as e:
            metadata["extraction_error"] = str(e)
        
        self.extracted_data.append(metadata)
        return metadata
    
    def _read_pdf_text(self, file_path: str) -> str:
        """
        Read text content from PDF file.
        
        Note: This is a simplified implementation. In production, use PyPDF2 or pdfplumber.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text from PDF (or sample text for demonstration)
        """
        try:
            # This is a placeholder - actual implementation would use:
            # from PyPDF2 import PdfReader
            # reader = PdfReader(file_path)
            # text = "".join([page.extract_text() for page in reader.pages])
            
            # For demonstration purposes, we return empty string
            # In production, PyPDF2 or pdfplumber would be used
            return ""
        except ImportError:
            return ""
    
    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text.
        
        Args:
            text: Text to search for emails
            
        Returns:
            List of unique email addresses found
        """
        emails = set()
        for pattern in self.EMAIL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            emails.update(matches)
        return list(emails)
    
    def extract_versions(self, text: str) -> List[str]:
        """
        Extract revision and version information.
        
        Args:
            text: Text to search for version info
            
        Returns:
            List of revision/version strings found
        """
        versions = set()
        for pattern in self.VERSION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            versions.update(matches)
        return list(versions)
    
    def extract_part_numbers(self, text: str) -> List[str]:
        """
        Extract semiconductor part numbers.
        
        Args:
            text: Text to search for part numbers
            
        Returns:
            List of part numbers found
        """
        pattern = self.TECHNICAL_PATTERNS["part_number"]
        part_numbers = re.findall(pattern, text)
        return list(set(part_numbers))
    
    def extract_technical_specs(self, text: str) -> Dict[str, List[str]]:
        """
        Extract technical specifications from text.
        
        Args:
            text: Text to parse
            
        Returns:
            Dictionary of technical specifications
        """
        specs = {}
        for spec_type, pattern in self.TECHNICAL_PATTERNS.items():
            if spec_type != "part_number":  # Already handled separately
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    specs[spec_type] = list(set(matches))
        return specs
    
    def batch_extract(self, directory: str) -> List[Dict[str, any]]:
        """
        Extract metadata from all PDFs in a directory.
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of metadata dictionaries
        """
        results = []
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_file in pdf_files:
            try:
                metadata = self.extract_metadata(str(pdf_file))
                results.append(metadata)
                print(f"✓ Processed: {pdf_file.name}")
            except Exception as e:
                print(f"✗ Error processing {pdf_file.name}: {e}")
        
        return results
    
    def generate_csv_report(self, output_file: str = "extracted_meta.csv"):
        """
        Generate a CSV report of extracted metadata.
        
        Args:
            output_file: Path to output CSV file
        """
        try:
            import csv
            
            if not self.extracted_data:
                print("No extracted data to export.")
                return
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                headers = [
                    "Filename",
                    "File Size (KB)",
                    "Modified Date",
                    "Part Numbers",
                    "Emails Found",
                    "Revisions",
                    "Package Types",
                    "Frequencies"
                ]
                writer.writerow(headers)
                
                # Write data rows
                for entry in self.extracted_data:
                    writer.writerow([
                        entry.get("filename", ""),
                        f"{entry.get('file_size_kb', 0):.2f}",
                        entry.get("modified_date", ""),
                        "; ".join(entry.get("part_numbers", [])),
                        "; ".join(entry.get("emails", [])),
                        "; ".join(entry.get("revisions", [])),
                        "; ".join(entry.get("technical_specs", {}).get("package_type", [])),
                        "; ".join(entry.get("technical_specs", {}).get("frequency", [])),
                    ])
            
            print(f"✓ Report exported to {output_file}")
        except ImportError:
            print("CSV export requires standard library (csv) - already available")


class EmailIntelligenceExtractor:
    """
    Specialized extractor for identifying technical contacts and support paths.
    """
    
    # STMicroelectronics specific contact patterns
    ST_SUPPORT_PATTERNS = {
        "support": r'support@st\.com',
        "technical": r'(?:technical|tech)[\._-]support@st\.com',
        "errata": r'(?:errata|updates)@st\.com',
        "applications": r'applications@st\.com',
        "general": r'(?:info|contact)@st\.com',
    }
    
    def __init__(self):
        """Initialize the Email Intelligence Extractor."""
        self.contacts_found = []
    
    def identify_support_channels(self, text: str) -> Dict[str, List[str]]:
        """
        Identify ST Microelectronics support channels from text.
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary mapping contact types to email addresses
        """
        channels = {}
        for channel_type, pattern in self.ST_SUPPORT_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                channels[channel_type] = list(set(matches))
                self.contacts_found.extend(matches)
        return channels
    
    def extract_contact_info(self, text: str) -> Dict[str, any]:
        """
        Extract comprehensive contact information.
        
        Args:
            text: Text to parse
            
        Returns:
            Dictionary with contact details
        """
        # Extract phone numbers
        phone_pattern = r'\+?[\d\s\-\(\)]{10,}'
        phones = re.findall(phone_pattern, text)
        
        # Extract web addresses
        web_pattern = r'(?:www\.)?(?:[a-z0-9\-]+\.)+[a-z]{2,}'
        websites = re.findall(web_pattern, text, re.IGNORECASE)
        
        return {
            "emails": EmailIntelligenceExtractor._extract_all_emails(text),
            "phones": list(set(phones)),
            "websites": list(set(websites)),
            "support_channels": self.identify_support_channels(text),
        }
    
    @staticmethod
    def _extract_all_emails(text: str) -> List[str]:
        """Extract all email addresses from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return list(set(re.findall(pattern, text)))


def main():
    """Example usage of the PDF Parser."""
    print("=" * 60)
    print("PDF METADATA EXTRACTOR - EXAMPLE USAGE")
    print("=" * 60)
    
    # Example 1: Extract metadata from sample PDF
    extractor = PDFMetadataExtractor()
    
    # Simulate text from a typical datasheet
    sample_datasheet_text = """
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
    
    For more information, visit www.st.com
    Phone: +1-800-ST-MICRO
    """
    
    print("\n1. Extracting emails from sample datasheet text:")
    emails = extractor.extract_emails(sample_datasheet_text)
    print(f"   Found {len(emails)} email(s):")
    for email in emails:
        print(f"   - {email}")
    
    print("\n2. Extracting version information:")
    versions = extractor.extract_versions(sample_datasheet_text)
    print(f"   Found {len(versions)} version(s):")
    for version in versions:
        print(f"   - {version}")
    
    print("\n3. Extracting part numbers:")
    parts = extractor.extract_part_numbers(sample_datasheet_text)
    print(f"   Found {len(parts)} part number(s):")
    for part in parts:
        print(f"   - {part}")
    
    print("\n4. Extracting technical specifications:")
    specs = extractor.extract_technical_specs(sample_datasheet_text)
    for spec_type, values in specs.items():
        print(f"   {spec_type}: {', '.join(values)}")
    
    # Example 2: Email Intelligence Extraction
    print("\n" + "=" * 60)
    print("EMAIL INTELLIGENCE EXTRACTION")
    print("=" * 60)
    
    email_extractor = EmailIntelligenceExtractor()
    contact_info = email_extractor.extract_contact_info(sample_datasheet_text)
    
    print("\nContact Information Found:")
    print(f"  Emails: {contact_info['emails']}")
    print(f"  Phones: {contact_info['phones']}")
    print(f"  Websites: {contact_info['websites']}")
    print(f"\nSupport Channels:")
    for channel, addresses in contact_info['support_channels'].items():
        print(f"  {channel}: {addresses}")


if __name__ == "__main__":
    main()
