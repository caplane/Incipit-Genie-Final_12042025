"""
incipit_extractor.py

Punctuation-Based Semantic Extraction for Incipit Notes

This module implements the core insight that authors encode semantic boundaries
through their punctuation choices. Rather than arbitrary word counts, we use
punctuation as a hierarchy of boundary indicators:

    1. Period (.)      → Complete thought boundary
    2. Semicolon (;)   → Independent clause boundary  
    3. Colon (:)       → Elaboration boundary
    4. Comma (,)       → Phrase boundary

The extraction finds where the current thought unit BEGINS (after a major
punctuation boundary), then takes the first few words as the incipit.
"""

import re
from typing import Optional, Tuple


class IncipitExtractor:
    """
    Extracts semantically meaningful incipit phrases from text using
    punctuation-based boundary detection.
    """
    
    # Punctuation hierarchy (higher = stronger boundary)
    BOUNDARY_HIERARCHY = {
        '.': 4,   # Complete thought
        ';': 3,   # Independent clause
        ':': 2,   # Elaboration point
    }
    
    # Secondary boundaries (used if no primary boundary found)
    SECONDARY_BOUNDARIES = {
        ',': 1,   # Phrase boundary
    }
    
    # Target incipit length range
    MIN_WORDS = 2
    MAX_WORDS = 8
    DEFAULT_WORDS = 5  # Fallback word count when no boundary found
    
    def __init__(self, word_count: int = None, used_incipits: set = None):
        """
        Initialize the extractor.
        
        Args:
            word_count: Number of words to extract for incipit (3-8). Default is 3.
            used_incipits: Set of incipits already used (to avoid duplicates).
        """
        # Set the target word count (clamp to valid range)
        if word_count is None:
            self.target_words = self.DEFAULT_WORDS
        else:
            self.target_words = max(self.MIN_WORDS, min(self.MAX_WORDS, word_count))
        
        # Track used incipits to avoid duplicates
        self.used_incipits = used_incipits or set()
        
        # Patterns for special boundary markers
        # Legal case pattern: captures "Party1 v. Party2" where Party2 can be multiple words
        self.legal_case_pattern = re.compile(r'\b([A-Z][a-zA-Z]*)\s+v\.\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)', re.IGNORECASE)
        
    def extract_incipit(self, text_before_ref: str, text_after_ref: str = "") -> str:
        """
        Extract an incipit phrase from the text preceding a reference marker.
        
        Args:
            text_before_ref: Text immediately before the superscript/reference
            text_after_ref: Text immediately after (for context, rarely used)
            
        Returns:
            The extracted incipit phrase (without trailing colon)
        """
        if not text_before_ref or not text_before_ref.strip():
            return "See"
        
        # Clean and normalize the text
        text = self._normalize_text(text_before_ref)
        
        # Get context (use full text - no artificial truncation)
        context = text
        
        # Try extraction strategies in order of preference
        incipit = (
            self._try_quote_extraction(context) or
            self._try_legal_case_extraction(context) or
            self._try_thought_unit_extraction(context) or
            self._fallback_extraction(context)
        )
        
        return self._finalize_incipit(incipit)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text."""
        # Normalize various quote types (using unicode escapes to avoid regex issues)
        text = text.replace('\u201c', '"').replace('\u201d', '"')  # curly double quotes
        text = text.replace('\u2018', "'").replace('\u2019', "'")  # curly single quotes
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _try_quote_extraction(self, context: str) -> Optional[str]:
        """
        Extract incipit based on quoted phrases near the endnote.
        
        Rules:
        1. Look for double-quoted phrase ending within ~150 chars of endnote
        2. Special case: if text ends with "quote"—[clause], look back past em-dash
        3. If split quote pattern ("...," [words], "..."), use first segment
        4. If lead-in ≤ 4 words, include lead-in + quote
        5. If lead-in > 4 words, use quote alone
        6. If quote ≤ 2 words, always include lead-in
        """
        # Check for em-dash pattern: "quote"—[clause] at end
        # Only trigger if em-dash clause ends near the endnote position
        # Look for em-dash in the last 250 chars
        tail = context[-250:] if len(context) > 250 else context
        em_dash_pos_in_tail = tail.rfind('—')
        
        # Only proceed if em-dash exists and there's text AFTER it (the clause)
        # and the clause is relatively short (< 100 chars after em-dash)
        if em_dash_pos_in_tail > 0:
            text_after_dash_in_tail = tail[em_dash_pos_in_tail + 1:]
            # The clause after em-dash should be short (this is what precedes the endnote)
            if len(text_after_dash_in_tail) < 100:
                # Map back to full context
                em_dash_pos = len(context) - len(tail) + em_dash_pos_in_tail
                
                # Check if there's a closing quote just before the em-dash
                text_before_dash = context[:em_dash_pos].rstrip()
                if text_before_dash.endswith('"'):
                    # Found pattern: ..."—[clause]. Use text before em-dash for quote search
                    search_region = text_before_dash[-150:] if len(text_before_dash) > 150 else text_before_dash
                    close_quote_pos_in_region = search_region.rfind('"')
                    if close_quote_pos_in_region >= 0:
                        # Map back to full context
                        offset = len(text_before_dash) - len(search_region)
                        close_quote_pos = offset + close_quote_pos_in_region
                        
                        # Find opening quote
                        open_quote_pos = text_before_dash.rfind('"', 0, close_quote_pos)
                        if open_quote_pos >= 0:
                            quoted_text = text_before_dash[open_quote_pos + 1:close_quote_pos].strip()
                            if quoted_text and len(quoted_text.split()) >= 2:
                                # Find lead-in
                                sentence_start = 0
                                period_pattern = re.compile(r'[.!?]["\']*\s+')
                                matches = list(period_pattern.finditer(text_before_dash[:open_quote_pos]))
                                if matches:
                                    sentence_start = matches[-1].end()
                                
                                lead_in = text_before_dash[sentence_start:open_quote_pos].strip()
                                lead_in_words = len(lead_in.split()) if lead_in else 0
                                quote_words = len(quoted_text.split())
                                
                                # Apply lead-in rules
                                if quote_words <= 2 and lead_in:
                                    return lead_in + ' "' + quoted_text + '"'
                                elif lead_in_words <= 4 and lead_in:
                                    return lead_in + ' "' + quoted_text + '"'
                                else:
                                    return '"' + quoted_text + '"'
        
        # Standard approach: look for quote in last 150 chars
        search_region = context[-150:] if len(context) > 150 else context
        
        # Find the last closing quote in the search region
        last_close_quote = search_region.rfind('"')
        if last_close_quote == -1:
            return None
        
        # Map position back to full context
        offset = len(context) - len(search_region)
        close_quote_pos = offset + last_close_quote
        
        # Check for split quote pattern: "...," [1-5 words], "..."
        split_pattern = re.compile(r'"([^"]+)" (\w+(?:\s+\w+){0,4}),? "([^"]+)"')
        for match in split_pattern.finditer(context):
            if match.end() >= len(context) - 50:
                first_segment = match.group(1).strip().rstrip(',')
                if len(first_segment.split()) >= 2:
                    return '"' + first_segment + '"'
        
        # Find the opening quote
        open_quote_pos = context.rfind('"', 0, close_quote_pos)
        if open_quote_pos == -1:
            return None
        
        quoted_text = context[open_quote_pos + 1:close_quote_pos].strip()
        if not quoted_text:
            return None
        
        quote_word_count = len(quoted_text.split())
        
        # Find sentence start for lead-in calculation
        sentence_start_pos = 0
        period_pattern = re.compile(r'[.!?]["\']*\s+')
        matches = list(period_pattern.finditer(context[:open_quote_pos]))
        if matches:
            sentence_start_pos = matches[-1].end()
        
        lead_in = context[sentence_start_pos:open_quote_pos].strip()
        lead_in_word_count = len(lead_in.split()) if lead_in else 0
        
        # Apply rules
        if quote_word_count <= 2 and lead_in:
            return lead_in + ' "' + quoted_text + '"'
        
        if lead_in_word_count <= 4 and lead_in:
            return lead_in + ' "' + quoted_text + '"'
        else:
            return '"' + quoted_text + '"'
    
    def _try_legal_case_extraction(self, context: str) -> Optional[str]:
        """
        Extract legal case citations (e.g., "Osheroff v. Chestnut Lodge").
        The "v." marker serves as a strong semantic boundary.
        """
        match = self.legal_case_pattern.search(context)
        if match:
            # Check if the case name is near the end (within last 80 chars)
            if match.end() > len(context) - 80:
                case_name = match.group(0)
                # Clean up: remove trailing words that are likely not part of the case name
                # Common verbs/words that follow case names
                stop_words = ['established', 'held', 'ruled', 'decided', 'found', 
                              'determined', 'concluded', 'stated', 'set', 'made',
                              'is', 'was', 'has', 'had', 'which', 'that', 'where']
                words = case_name.split()
                cleaned_words = []
                for word in words:
                    # Stop if we hit a common verb (lowercase check after v.)
                    if word.lower() in stop_words:
                        break
                    cleaned_words.append(word)
                return ' '.join(cleaned_words) if cleaned_words else None
        return None
    
    def _try_thought_unit_extraction(self, context: str) -> Optional[str]:
        """
        Find where the current SENTENCE begins, then extract forward to the first
        natural semantic boundary.
        
        Algorithm:
        1. Find sentence start (previous period or start of text)
        2. From sentence start, extract FORWARD to first semantic break:
           - Comma (,)
           - Colon (:)
           - Em-dash (—)
           - Period (.) for short declarative statements
        3. Return that opening phrase as the incipit
        """
        # Step 1: Find sentence start (look for last period before the endnote)
        # We want the period that ENDS the previous sentence
        sentence_start_pos = 0
        
        # Find the last period (which ends the previous sentence)
        # Look for period followed by optional closing punctuation, space, and capital letter
        # Handles: ". The", "." The", ".' The", ".) The"
        period_pattern = re.compile(r'\.["\'""'')]*\s+([A-Z])')
        matches = list(period_pattern.finditer(context))
        
        if matches:
            # Get the last match - this is where our sentence starts
            last_match = matches[-1]
            # Find where the capital letter starts (the captured group)
            sentence_start_pos = last_match.start(1)
        else:
            # No clear sentence boundary - start from beginning
            sentence_start_pos = 0
        
        # Extract the sentence from start to end of context
        sentence = context[sentence_start_pos:].strip()
        
        if not sentence:
            return None
        
        # Step 2: From sentence start, find first semantic boundary
        # Order matters: look for these breaks in sequence
        
        # Em-dash (—) is a strong semantic break
        em_dash_pos = sentence.find('—')
        
        # Colon is a strong break
        colon_pos = sentence.find(':')
        
        # Comma is a softer break
        comma_pos = sentence.find(',')
        
        # Period within the sentence (for short declarative statements)
        # But NOT the final period which is the sentence terminator
        period_pos = sentence.find('.')
        # If period is near the end (within last 5 chars), it's the sentence terminator
        if period_pos > 0 and period_pos > len(sentence) - 5:
            period_pos = -1  # Ignore final period
        
        # Find the earliest semantic boundary
        boundaries = []
        if em_dash_pos > 0:
            boundaries.append(('em_dash', em_dash_pos))
        if colon_pos > 0:
            boundaries.append(('colon', colon_pos))
        if comma_pos > 0:
            boundaries.append(('comma', comma_pos))
        if period_pos > 0:
            boundaries.append(('period', period_pos))
        
        if boundaries:
            # Sort by position and take the earliest
            boundaries.sort(key=lambda x: x[1])
            
            for boundary_type, boundary_pos in boundaries:
                # Extract up to (but not including) the boundary
                incipit = sentence[:boundary_pos].strip()
                
                # Determine minimum words based on boundary type
                # Commas tend to produce fragments; need 3 words minimum
                # Colons, em-dashes, periods produce complete thoughts; 2 words OK
                if boundary_type == 'comma':
                    min_words_required = 3
                else:
                    min_words_required = self.MIN_WORDS  # 2
                
                # Validate word count
                word_count = len(incipit.split())
                if word_count >= min_words_required:
                    # Check for duplicate
                    if self._is_duplicate(incipit):
                        # Try next boundary instead
                        continue
                    return incipit
                # Otherwise, continue to next boundary
        
        # No valid boundary found - take first N words
        fallback = self._extract_first_words(sentence)
        
        # If fallback is also a duplicate, try previous sentence
        if self._is_duplicate(fallback) and matches and len(matches) > 1:
            # Try the second-to-last sentence
            prev_match = matches[-2]
            prev_sentence_start = prev_match.start() + 2
            prev_sentence_end = matches[-1].start()
            prev_sentence = context[prev_sentence_start:prev_sentence_end].strip()
            if prev_sentence:
                alt_incipit = self._extract_first_words(prev_sentence)
                if not self._is_duplicate(alt_incipit):
                    return alt_incipit
        
        return fallback
    
    def _is_duplicate(self, incipit: str) -> bool:
        """Check if this incipit (or a similar one) has already been used."""
        if not self.used_incipits:
            return False
        
        # Normalize for comparison
        normalized = self._normalize_for_comparison(incipit)
        
        for used in self.used_incipits:
            used_normalized = self._normalize_for_comparison(used)
            
            # Exact match
            if normalized == used_normalized:
                return True
            
            # One contains the other (catches partial matches)
            if normalized in used_normalized or used_normalized in normalized:
                return True
            
            # Same first 3 words (catches "Slingshot AI cites" vs "Shot AI cites")
            norm_words = normalized.split()[:3]
            used_words = used_normalized.split()[:3]
            if len(norm_words) >= 3 and len(used_words) >= 3:
                if norm_words == used_words:
                    return True
        
        return False
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate comparison."""
        # Lowercase, strip punctuation, normalize whitespace
        import string
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    def _extract_first_words(self, text: str, max_words: int = None) -> str:
        """Extract the first N words from text, respecting minimum word counts."""
        if max_words is None:
            max_words = self.target_words
        
        words = text.split()
        
        if len(words) <= max_words:
            return text.strip()
        
        # Try to find a natural break within the first max_words+2
        candidate_words = words[:max_words + 2]
        candidate_text = ' '.join(candidate_words)
        
        # Check for comma within reasonable range
        # But only break at comma if we have at least 3 words before it
        comma_pos = candidate_text.find(',')
        if comma_pos > 0:
            words_before_comma = candidate_text[:comma_pos].split()
            if len(words_before_comma) >= 3 and comma_pos < len(' '.join(words[:max_words])):
                return candidate_text[:comma_pos].strip()
        
        # Otherwise just take first max_words
        return ' '.join(words[:max_words])
    
    def _fallback_extraction(self, context: str) -> str:
        """Fallback: just take the first few words of the context."""
        words = context.split()
        if len(words) <= self.target_words:
            return context.strip()
        return ' '.join(words[:self.target_words])
    
    def _finalize_incipit(self, incipit: str) -> str:
        """Clean up and finalize the incipit phrase."""
        if not incipit:
            return "See"
        
        # Remove leading punctuation and whitespace
        incipit = re.sub(r'^[\s\.,;:]+', '', incipit)
        
        # Remove trailing punctuation (we'll add colon later)
        incipit = re.sub(r'[\s\.,;:]+$', '', incipit)
        
        # Capitalize first letter
        if incipit and incipit[0].islower():
            incipit = incipit[0].upper() + incipit[1:]
        
        # Final sanity check
        if not incipit or len(incipit) < 2:
            return "See"
        
        return incipit


def extract_incipit(text_before_ref: str, text_after_ref: str = "", word_count: int = None, used_incipits: set = None) -> str:
    """
    Convenience function for extracting incipits.
    
    Args:
        text_before_ref: Text immediately before the reference marker
        text_after_ref: Text immediately after (optional context)
        word_count: Number of words to extract (3-8). Default is 3.
        used_incipits: Set of incipits already used (to avoid duplicates).
        
    Returns:
        The extracted incipit phrase
    """
    extractor = IncipitExtractor(word_count=word_count, used_incipits=used_incipits)
    return extractor.extract_incipit(text_before_ref, text_after_ref)
