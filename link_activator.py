"""
link_activator.py

Post-processing module that converts plain text URLs in Word documents
into clickable hyperlinks.

Processes document.xml, endnotes.xml, and footnotes.xml to find URLs
and convert them to proper Word HYPERLINK field codes with blue
underlined styling.
"""

import os
import re
import html
import zipfile
import tempfile
import shutil
from io import BytesIO


class LinkActivator:
    """
    Post-processing module that converts plain text URLs in Word documents
    into clickable hyperlinks.
    """
    
    # Pattern to match URLs
    URL_PATTERN = re.compile(r'(https?://[^\s<>"]+)')
    
    @classmethod
    def process(cls, docx_bytes: bytes) -> bytes:
        """
        Process a .docx file to make all URLs clickable.
        
        Args:
            docx_bytes: Raw bytes of the input .docx file
            
        Returns:
            Bytes of the processed .docx file with clickable URLs
        """
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract docx to temp directory
            with zipfile.ZipFile(BytesIO(docx_bytes), 'r') as zf:
                zf.extractall(temp_dir)
            
            # Process each relevant XML file
            target_files = [
                'word/document.xml',
                'word/endnotes.xml',
                'word/footnotes.xml'
            ]
            
            for xml_file in target_files:
                full_path = os.path.join(temp_dir, xml_file)
                if os.path.exists(full_path):
                    cls._process_xml_file(full_path)
            
            # Repackage as docx
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname)
            
            buffer.seek(0)
            return buffer.read()
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    @classmethod
    def _process_xml_file(cls, file_path: str):
        """Process a single XML file to convert URLs to hyperlinks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match <w:t> elements containing URLs
        # We need to be careful not to process URLs that are already hyperlinks
        
        def convert_url_in_text(match):
            """Convert a URL found in a w:t element to a hyperlink field."""
            full_match = match.group(0)
            prefix = match.group(1)  # Opening tag and any text before URL
            url = match.group(2)     # The URL
            suffix = match.group(3)  # Any text after URL and closing tag
            
            # Clean URL - remove trailing punctuation that's probably not part of URL
            clean_url = url.rstrip('.,;:)]\'"')
            trailing = url[len(clean_url):]
            
            # Escape URL for XML
            safe_url = html.escape(clean_url)
            
            # Build the HYPERLINK field structure
            hyperlink_xml = cls._build_hyperlink_field(safe_url, clean_url)
            
            # Reconstruct with any trailing punctuation
            if trailing:
                hyperlink_xml += f'<w:r><w:t>{html.escape(trailing)}</w:t></w:r>'
            
            return f'{prefix}{hyperlink_xml}{suffix}'
        
        # Pattern to find URLs within w:t elements
        # This is tricky because we need to:
        # 1. Find w:t elements containing URLs
        # 2. Not process URLs that are already in hyperlinks
        # 3. Preserve surrounding text
        
        # First, skip any content that's already a hyperlink
        # Look for w:t elements that contain URLs but aren't inside w:hyperlink
        
        # Simple approach: find all <w:t>...</w:t> containing URLs and replace
        pattern = r'(<w:t[^>]*>)([^<]*?)(https?://[^\s<>"]+)([^<]*?)(</w:t>)'
        
        def replace_url(match):
            t_open = match.group(1)
            text_before = match.group(2)
            url = match.group(3)
            text_after = match.group(4)
            t_close = match.group(5)
            
            # Check if this is inside a hyperlink (crude check)
            # Look backward in the content to see if we're in a hyperlink
            pos = match.start()
            context_before = content[max(0, pos-500):pos]
            
            # Count hyperlink opens vs closes before this point
            hyperlink_opens = context_before.count('<w:hyperlink')
            hyperlink_closes = context_before.count('</w:hyperlink>')
            
            # Also check for HYPERLINK in instrText (field-based hyperlinks)
            if 'HYPERLINK' in context_before[-200:]:
                return match.group(0)  # Already a hyperlink, skip
            
            if hyperlink_opens > hyperlink_closes:
                return match.group(0)  # Inside a hyperlink, skip
            
            # Clean URL
            clean_url = url.rstrip('.,;:)]\'"')
            trailing = url[len(clean_url):]
            safe_url = html.escape(clean_url)
            
            # Build result
            result = ""
            
            # Text before URL (if any)
            if text_before:
                result += f'{t_open}{text_before}{t_close}</w:r>'
            else:
                result += '</w:r>'
            
            # The hyperlink field
            result += cls._build_hyperlink_field(safe_url, clean_url)
            
            # Text after URL (if any), including any trailing punctuation
            after_text = trailing + text_after
            if after_text:
                result += f'<w:r>{t_open}{after_text}{t_close}'
            else:
                result += '<w:r>'
            
            return result
        
        # Apply the replacement
        # This is a simplified approach - may need refinement for edge cases
        new_content = re.sub(pattern, replace_url, content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    @classmethod
    def _build_hyperlink_field(cls, safe_url: str, display_text: str) -> str:
        """
        Build Word HYPERLINK field XML.
        
        Structure:
        - fldChar begin
        - instrText with HYPERLINK "url"
        - fldChar separate
        - Display text (blue, underlined)
        - fldChar end
        """
        # Escape display text for XML
        safe_display = html.escape(display_text)
        
        field_xml = (
            # Field begin
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            # Field instruction
            f'<w:r><w:instrText xml:space="preserve"> HYPERLINK "{safe_url}" </w:instrText></w:r>'
            # Field separator
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            # Display text with hyperlink styling (blue, underlined)
            f'<w:r><w:rPr><w:color w:val="0000FF"/><w:u w:val="single"/></w:rPr>'
            f'<w:t>{safe_display}</w:t></w:r>'
            # Field end
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        )
        
        return field_xml


def activate_links(docx_bytes: bytes) -> bytes:
    """
    Convenience function to activate links in a document.
    
    Args:
        docx_bytes: Raw bytes of the input .docx file
        
    Returns:
        Bytes of the processed .docx file with clickable URLs
    """
    return LinkActivator.process(docx_bytes)
