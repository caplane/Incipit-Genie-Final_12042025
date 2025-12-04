"""
document_processor.py

Direct XML manipulation of Word documents for incipit note conversion.

Updated: 2025-12-04 20:35 UTC

This module handles:
1. Extracting docx files (which are zip archives)
2. Parsing document.xml to find endnote references
3. Extracting text context around each reference
4. Removing superscript numbers from main body
5. Inserting bookmarks at incipit locations
6. Rewriting endnotes with PAGEREF fields + bold incipit + citation
7. Repackaging as docx
"""

import os
import re
import copy
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from io import BytesIO

from incipit_extractor import extract_incipit


# XML namespaces used in Word documents
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
}

# Register ALL common Word namespaces to preserve prefixes
ALL_NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
    'w16': 'http://schemas.microsoft.com/office/word/2018/wordml',
    'w16cex': 'http://schemas.microsoft.com/office/word/2018/wordml/cex',
    'w16cid': 'http://schemas.microsoft.com/office/word/2016/wordml/cid',
    'w16du': 'http://schemas.microsoft.com/office/word/2023/wordml/word16du',
    'w16sdtdh': 'http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash',
    'w16sdtfl': 'http://schemas.microsoft.com/office/word/2024/wordml/sdtformatlock',
    'w16se': 'http://schemas.microsoft.com/office/word/2015/wordml/symex',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
    'wpc': 'http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas',
    'wpg': 'http://schemas.microsoft.com/office/word/2010/wordprocessingGroup',
    'wpi': 'http://schemas.microsoft.com/office/word/2010/wordprocessingInk',
    'wne': 'http://schemas.microsoft.com/office/word/2006/wordml',
    'wp14': 'http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing',
    'w10': 'urn:schemas-microsoft-com:office:word',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'o': 'urn:schemas-microsoft-com:office:office',
    'v': 'urn:schemas-microsoft-com:vml',
    'cx': 'http://schemas.microsoft.com/office/drawing/2014/chartex',
    'cx1': 'http://schemas.microsoft.com/office/drawing/2015/9/8/chartex',
    'cx2': 'http://schemas.microsoft.com/office/drawing/2015/10/21/chartex',
    'cx3': 'http://schemas.microsoft.com/office/drawing/2016/5/9/chartex',
    'cx4': 'http://schemas.microsoft.com/office/drawing/2016/5/10/chartex',
    'cx5': 'http://schemas.microsoft.com/office/drawing/2016/5/11/chartex',
    'cx6': 'http://schemas.microsoft.com/office/drawing/2016/5/12/chartex',
    'cx7': 'http://schemas.microsoft.com/office/drawing/2016/5/13/chartex',
    'cx8': 'http://schemas.microsoft.com/office/drawing/2016/5/14/chartex',
    'aink': 'http://schemas.microsoft.com/office/drawing/2016/ink',
    'am3d': 'http://schemas.microsoft.com/office/drawing/2017/model3d',
    'oel': 'http://schemas.microsoft.com/office/2019/extlst',
}

# Register all namespaces BEFORE any parsing happens
for prefix, uri in ALL_NAMESPACES.items():
    ET.register_namespace(prefix, uri)


@dataclass
class EndnoteReference:
    """Represents an endnote reference found in the document."""
    note_id: str
    paragraph_index: int
    run_index: int
    text_before: str
    text_after: str
    incipit: str = ""
    bookmark_name: str = ""


@dataclass
class Endnote:
    """Represents an endnote's content."""
    note_id: str
    content: str  # Plain text (for fallback/debugging)
    content_runs: List[ET.Element] = None  # Original runs with formatting preserved
    xml_element: ET.Element = None


class DocumentProcessor:
    """
    Processes Word documents to convert endnotes to incipit format.
    Uses direct XML manipulation for precise control.
    """
    
    def __init__(self, docx_bytes: bytes, word_count: int = 3, format_style: str = 'bold'):
        """
        Initialize with the bytes of a .docx file.
        
        Args:
            docx_bytes: The raw bytes of the input .docx file
            word_count: Number of words for incipit phrase (3-8). Default is 3.
            format_style: 'bold' or 'italic' for incipit formatting. Default is 'bold'.
        """
        self.docx_bytes = docx_bytes
        self.word_count = max(3, min(8, word_count))  # Clamp to valid range
        self.format_style = format_style.lower() if format_style else 'bold'
        self.temp_dir = tempfile.mkdtemp()
        self.document_xml = None
        self.endnotes_xml = None
        self.references: List[EndnoteReference] = []
        self.endnotes: Dict[str, Endnote] = {}
        self.bookmark_counter = 1000  # Start high to avoid conflicts
        
    def process(self) -> bytes:
        """
        Main processing pipeline.
        
        Returns:
            Bytes of the transformed .docx file
        """
        try:
            # Step 1: Extract the docx
            self._extract_docx()
            
            # Step 2: Parse document.xml and endnotes.xml
            self._parse_xml_files()
            
            # Step 3: Find all endnote references and extract context
            self._find_endnote_references()
            
            # Step 4: Extract incipits for each reference
            self._extract_incipits()
            
            # Step 5: Read endnote content
            self._read_endnotes()
            
            # Step 6: Transform the document
            self._transform_document()
            
            # Step 7: Repackage and return
            return self._repackage_docx()
            
        finally:
            self.cleanup()
    
    def _extract_docx(self):
        """Extract the docx zip archive to temp directory."""
        with zipfile.ZipFile(BytesIO(self.docx_bytes), 'r') as zf:
            zf.extractall(self.temp_dir)
    
    def _parse_xml_files(self):
        """Parse the main XML files."""
        doc_path = os.path.join(self.temp_dir, 'word', 'document.xml')
        endnotes_path = os.path.join(self.temp_dir, 'word', 'endnotes.xml')
        
        if os.path.exists(doc_path):
            self.document_xml = ET.parse(doc_path)
        else:
            raise ValueError("No document.xml found in docx file")
        
        if os.path.exists(endnotes_path):
            self.endnotes_xml = ET.parse(endnotes_path)
        else:
            raise ValueError("No endnotes.xml found - document has no endnotes")
    
    def _find_endnote_references(self):
        """Find all endnote references in the document and extract surrounding text."""
        root = self.document_xml.getroot()
        body = root.find('.//w:body', NAMESPACES)
        
        if body is None:
            return
        
        paragraphs = body.findall('.//w:p', NAMESPACES)
        
        for para_idx, para in enumerate(paragraphs):
            # Get all text content and runs in this paragraph
            runs = para.findall('.//w:r', NAMESPACES)
            
            # Build text map: track position of each run's text
            full_text = ""
            run_positions = []  # (start_pos, end_pos, run_element)
            
            for run_idx, run in enumerate(runs):
                run_text = self._get_run_text(run)
                start = len(full_text)
                full_text += run_text
                end = len(full_text)
                run_positions.append((start, end, run, run_idx))
            
            # Find endnote references in this paragraph
            for run_idx, run in enumerate(runs):
                endnote_ref = run.find('.//w:endnoteReference', NAMESPACES)
                if endnote_ref is not None:
                    note_id = endnote_ref.get(f'{{{NAMESPACES["w"]}}}id')
                    if note_id and note_id not in ['0', '-1']:  # Skip separator notes
                        # Calculate text position of this run
                        run_start = sum(len(self._get_run_text(runs[i])) for i in range(run_idx))
                        
                        text_before = full_text[:run_start]
                        text_after = full_text[run_start:]
                        
                        # Also get text from previous paragraphs for more context
                        extended_before = self._get_extended_context_before(paragraphs, para_idx)
                        
                        ref = EndnoteReference(
                            note_id=note_id,
                            paragraph_index=para_idx,
                            run_index=run_idx,
                            text_before=extended_before + text_before,
                            text_after=text_after
                        )
                        self.references.append(ref)
    
    def _get_run_text(self, run: ET.Element) -> str:
        """Extract text content from a run element."""
        text = ""
        for t_elem in run.findall('.//w:t', NAMESPACES):
            if t_elem.text:
                text += t_elem.text
        return text
    
    def _get_extended_context_before(self, paragraphs: List[ET.Element], current_idx: int, 
                                      max_chars: int = 200) -> str:
        """Get text from previous paragraphs for extended context."""
        context = ""
        for i in range(current_idx - 1, max(0, current_idx - 5), -1):
            para_text = ""
            for run in paragraphs[i].findall('.//w:r', NAMESPACES):
                para_text += self._get_run_text(run)
            context = para_text + " " + context
            if len(context) >= max_chars:
                break
        return context[-max_chars:] if len(context) > max_chars else context
    
    def _extract_incipits(self):
        """Extract incipit phrases for each reference, avoiding duplicates."""
        used_incipits = set()
        
        for ref in self.references:
            ref.incipit = extract_incipit(
                ref.text_before, 
                ref.text_after, 
                word_count=self.word_count,
                used_incipits=used_incipits
            )
            ref.bookmark_name = f"_IncipitRef{ref.note_id}"
            
            # Add to used set to prevent duplicates in subsequent notes
            used_incipits.add(ref.incipit)
    
    def _read_endnotes(self):
        """Read content from each endnote, preserving original run formatting."""
        if self.endnotes_xml is None:
            return
        
        root = self.endnotes_xml.getroot()
        for endnote in root.findall('.//w:endnote', NAMESPACES):
            note_id = endnote.get(f'{{{NAMESPACES["w"]}}}id')
            if note_id and note_id not in ['0', '-1']:
                content, content_runs = self._extract_endnote_content_with_runs(endnote)
                self.endnotes[note_id] = Endnote(
                    note_id=note_id,
                    content=content,
                    content_runs=content_runs,
                    xml_element=endnote
                )
    
    def _extract_endnote_content(self, endnote: ET.Element) -> str:
        """Extract text content from an endnote, skipping the reference number."""
        content = ""
        for para in endnote.findall('.//w:p', NAMESPACES):
            para_text = ""
            for run in para.findall('.//w:r', NAMESPACES):
                # Skip runs that contain the endnote reference marker
                if run.find('.//w:endnoteRef', NAMESPACES) is not None:
                    continue
                para_text += self._get_run_text(run)
            content += para_text.strip() + " "
        return content.strip()
    
    def _extract_endnote_content_with_runs(self, endnote: ET.Element) -> Tuple[str, List[ET.Element]]:
        """
        Extract content from an endnote, preserving original run formatting.
        
        Returns:
            Tuple of (plain_text_content, list_of_run_elements)
        """
        content = ""
        content_runs = []
        
        for para in endnote.findall('.//w:p', NAMESPACES):
            for run in para.findall('.//w:r', NAMESPACES):
                # Skip runs that contain the endnote reference marker
                if run.find('.//w:endnoteRef', NAMESPACES) is not None:
                    continue
                
                # Deep copy the run to preserve formatting
                run_copy = copy.deepcopy(run)
                content_runs.append(run_copy)
                
                # Also extract plain text for fallback
                content += self._get_run_text(run)
        
        return content.strip(), content_runs
    
    def _transform_document(self):
        """Apply all transformations to create incipit notes."""
        # Transform main document: remove superscripts, add bookmarks
        self._transform_main_document()
        
        # Append notes as regular paragraphs at end of document
        self._append_notes_to_document()
        
        # Clear the endnotes.xml (keep separators only)
        self._clear_endnotes()
        
        # Save modified XML files
        self._save_xml_files()
    
    def _append_notes_to_document(self):
        """Append incipit notes as regular paragraphs at end of document."""
        root = self.document_xml.getroot()
        body = root.find('.//w:body', NAMESPACES)
        
        # Find sectPr (section properties) - notes go before this
        sectPr = body.find('w:sectPr', NAMESPACES)
        
        # Add a page break before notes section
        page_break_para = self._create_page_break_paragraph()
        if sectPr is not None:
            body.insert(list(body).index(sectPr), page_break_para)
        else:
            body.append(page_break_para)
        
        # Add "Notes" heading
        notes_heading = self._create_notes_heading()
        if sectPr is not None:
            body.insert(list(body).index(sectPr), notes_heading)
        else:
            body.append(notes_heading)
        
        # Add each note as a paragraph (in order by note_id)
        sorted_refs = sorted(self.references, key=lambda r: int(r.note_id))
        
        for ref in sorted_refs:
            if ref.note_id not in self.endnotes:
                continue
            
            endnote = self.endnotes[ref.note_id]
            note_para = self._create_incipit_paragraph(
                ref.bookmark_name,
                ref.incipit,
                endnote.content,
                endnote.content_runs
            )
            
            if sectPr is not None:
                body.insert(list(body).index(sectPr), note_para)
            else:
                body.append(note_para)
    
    def _create_page_break_paragraph(self) -> ET.Element:
        """Create a paragraph with a page break."""
        w = NAMESPACES['w']
        para = ET.Element(f'{{{w}}}p')
        run = ET.SubElement(para, f'{{{w}}}r')
        br = ET.SubElement(run, f'{{{w}}}br')
        br.set(f'{{{w}}}type', 'page')
        return para
    
    def _create_notes_heading(self) -> ET.Element:
        """Create a 'Notes' heading paragraph."""
        w = NAMESPACES['w']
        para = ET.Element(f'{{{w}}}p')
        
        # Paragraph properties for heading style
        pPr = ET.SubElement(para, f'{{{w}}}pPr')
        pStyle = ET.SubElement(pPr, f'{{{w}}}pStyle')
        pStyle.set(f'{{{w}}}val', 'Heading1')
        
        # The text "Notes"
        run = ET.SubElement(para, f'{{{w}}}r')
        text = ET.SubElement(run, f'{{{w}}}t')
        text.text = 'Notes'
        
        return para
    
    def _clear_endnotes(self):
        """Clear endnotes.xml, keeping only separator entries."""
        if self.endnotes_xml is None:
            return
        
        root = self.endnotes_xml.getroot()
        
        # Remove all endnotes except separators (id -1 and 0)
        for endnote in list(root.findall('w:endnote', NAMESPACES)):
            note_id = endnote.get(f'{{{NAMESPACES["w"]}}}id')
            if note_id not in ['-1', '0']:
                root.remove(endnote)
    
    def _transform_main_document(self):
        """Remove superscript refs and insert bookmarks in main document."""
        root = self.document_xml.getroot()
        body = root.find('.//w:body', NAMESPACES)
        paragraphs = body.findall('.//w:p', NAMESPACES)
        
        # Process references in reverse order to maintain indices
        for ref in reversed(self.references):
            para = paragraphs[ref.paragraph_index]
            runs = para.findall('.//w:r', NAMESPACES)
            
            if ref.run_index < len(runs):
                run = runs[ref.run_index]
                
                # Insert bookmark before this run
                self._insert_bookmark_before_run(para, run, ref.bookmark_name)
                
                # Remove the endnote reference from this run
                self._remove_endnote_ref_from_run(run)
    
    def _insert_bookmark_before_run(self, para: ET.Element, run: ET.Element, 
                                     bookmark_name: str):
        """Insert a bookmark start/end pair before the specified run."""
        # Find the index of this run in the paragraph
        run_index = None
        for i, child in enumerate(para):
            if child == run:
                run_index = i
                break
        
        if run_index is None:
            return
        
        # Create bookmark elements
        bookmark_id = str(self.bookmark_counter)
        self.bookmark_counter += 1
        
        bookmark_start = ET.Element(f'{{{NAMESPACES["w"]}}}bookmarkStart')
        bookmark_start.set(f'{{{NAMESPACES["w"]}}}id', bookmark_id)
        bookmark_start.set(f'{{{NAMESPACES["w"]}}}name', bookmark_name)
        
        bookmark_end = ET.Element(f'{{{NAMESPACES["w"]}}}bookmarkEnd')
        bookmark_end.set(f'{{{NAMESPACES["w"]}}}id', bookmark_id)
        
        # Insert bookmark start before the run, bookmark end after
        para.insert(run_index, bookmark_start)
        para.insert(run_index + 2, bookmark_end)  # +2 because we just inserted one element
    
    def _remove_endnote_ref_from_run(self, run: ET.Element):
        """Remove the endnoteReference element from a run."""
        # Find and remove endnoteReference
        for elem in run.findall('.//w:endnoteReference', NAMESPACES):
            parent = self._find_parent(run, elem)
            if parent is not None:
                parent.remove(elem)
        
        # If run is now empty (only contained the reference), we could remove it
        # But safer to leave it as an empty run
    
    def _find_parent(self, root: ET.Element, target: ET.Element) -> Optional[ET.Element]:
        """Find the parent of a target element."""
        for parent in root.iter():
            for child in parent:
                if child == target:
                    return parent
        return None
    
    def _transform_endnotes(self):
        """Transform endnotes to incipit format with PAGEREF fields."""
        if self.endnotes_xml is None:
            return
        
        root = self.endnotes_xml.getroot()
        
        for ref in self.references:
            if ref.note_id not in self.endnotes:
                continue
            
            endnote = self.endnotes[ref.note_id]
            endnote_elem = endnote.xml_element
            
            # Clear existing content
            for child in list(endnote_elem):
                endnote_elem.remove(child)
            
            # Build new content: PAGEREF + bold incipit + colon + original citation
            new_para = self._create_incipit_paragraph(
                ref.bookmark_name,
                ref.incipit,
                endnote.content,
                endnote.content_runs
            )
            endnote_elem.append(new_para)
    
    def _create_incipit_paragraph(self, bookmark_name: str, incipit: str, 
                                   citation: str, content_runs: List[ET.Element] = None) -> ET.Element:
        """
        Create a paragraph with: italic PAGEREF + bold incipit + colon + citation
        
        Args:
            content_runs: Original run elements with formatting preserved (if available)
        """
        w = NAMESPACES['w']
        
        # Determine formatting element for incipit (bold or italic)
        format_tag = f'{{{w}}}b' if self.format_style == 'bold' else f'{{{w}}}i'
        
        # Create paragraph
        para = ET.Element(f'{{{w}}}p')
        
        # --- PAGEREF field (italic) ---
        # Field begin
        run_begin = ET.SubElement(para, f'{{{w}}}r')
        rpr_begin = ET.SubElement(run_begin, f'{{{w}}}rPr')
        ET.SubElement(rpr_begin, f'{{{w}}}i')  # Italic
        fld_begin = ET.SubElement(run_begin, f'{{{w}}}fldChar')
        fld_begin.set(f'{{{w}}}fldCharType', 'begin')
        
        # Field instruction
        run_instr = ET.SubElement(para, f'{{{w}}}r')
        rpr_instr = ET.SubElement(run_instr, f'{{{w}}}rPr')
        ET.SubElement(rpr_instr, f'{{{w}}}i')  # Italic
        instr_text = ET.SubElement(run_instr, f'{{{w}}}instrText')
        instr_text.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        instr_text.text = f' PAGEREF {bookmark_name} \\h '
        
        # Field separator
        run_sep = ET.SubElement(para, f'{{{w}}}r')
        rpr_sep = ET.SubElement(run_sep, f'{{{w}}}rPr')
        ET.SubElement(rpr_sep, f'{{{w}}}i')  # Italic
        fld_sep = ET.SubElement(run_sep, f'{{{w}}}fldChar')
        fld_sep.set(f'{{{w}}}fldCharType', 'separate')
        
        # Field result (placeholder - will be updated by Word)
        run_result = ET.SubElement(para, f'{{{w}}}r')
        rpr_result = ET.SubElement(run_result, f'{{{w}}}rPr')
        ET.SubElement(rpr_result, f'{{{w}}}i')  # Italic
        result_text = ET.SubElement(run_result, f'{{{w}}}t')
        result_text.text = "##"  # Placeholder
        
        # Field end
        run_end = ET.SubElement(para, f'{{{w}}}r')
        rpr_end = ET.SubElement(run_end, f'{{{w}}}rPr')
        ET.SubElement(rpr_end, f'{{{w}}}i')  # Italic
        fld_end = ET.SubElement(run_end, f'{{{w}}}fldChar')
        fld_end.set(f'{{{w}}}fldCharType', 'end')
        
        # --- Period and space after page number ---
        run_period = ET.SubElement(para, f'{{{w}}}r')
        period_text = ET.SubElement(run_period, f'{{{w}}}t')
        period_text.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        period_text.text = "  "  # TWO spaces (no period)
        
        # --- Incipit phrase (bold or italic based on format_style) ---
        run_incipit = ET.SubElement(para, f'{{{w}}}r')
        rpr_incipit = ET.SubElement(run_incipit, f'{{{w}}}rPr')
        ET.SubElement(rpr_incipit, format_tag)  # Bold or Italic
        incipit_text = ET.SubElement(run_incipit, f'{{{w}}}t')
        incipit_text.text = incipit
        
        # --- Colon and space (same formatting as incipit) ---
        run_colon = ET.SubElement(para, f'{{{w}}}r')
        rpr_colon = ET.SubElement(run_colon, f'{{{w}}}rPr')
        ET.SubElement(rpr_colon, format_tag)  # Bold or Italic
        colon_text = ET.SubElement(run_colon, f'{{{w}}}t')
        colon_text.text = ": "
        
        # --- Citation content (with original formatting preserved) ---
        if content_runs:
            # Append deep copies of original runs to preserve italics, etc.
            for run in content_runs:
                para.append(copy.deepcopy(run))
        else:
            # Fallback to plain text
            run_citation = ET.SubElement(para, f'{{{w}}}r')
            citation_text = ET.SubElement(run_citation, f'{{{w}}}t')
            citation_text.text = citation
        
        return para
    
    def _save_xml_files(self):
        """Save modified XML files back to the temp directory."""
        doc_path = os.path.join(self.temp_dir, 'word', 'document.xml')
        endnotes_path = os.path.join(self.temp_dir, 'word', 'endnotes.xml')
        
        # Write with XML declaration
        self.document_xml.write(doc_path, encoding='UTF-8', xml_declaration=True)
        if self.endnotes_xml:
            self.endnotes_xml.write(endnotes_path, encoding='UTF-8', xml_declaration=True)
    
    def _repackage_docx(self) -> bytes:
        """Repackage the temp directory as a docx file."""
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zf.write(file_path, arcname)
        buffer.seek(0)
        return buffer.read()
    
    def cleanup(self):
        """Remove temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def process_document(docx_bytes: bytes, word_count: int = 3, format_style: str = 'bold') -> bytes:
    """
    Convenience function to process a document.
    
    Args:
        docx_bytes: Raw bytes of the input .docx file
        word_count: Number of words for incipit phrase (3-8). Default is 3.
        format_style: 'bold' or 'italic' for incipit formatting. Default is 'bold'.
        
    Returns:
        Bytes of the transformed .docx file
    """
    processor = DocumentProcessor(docx_bytes, word_count=word_count, format_style=format_style)
    return processor.process()
