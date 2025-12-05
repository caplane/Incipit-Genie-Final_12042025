"""
citeflex/claude_router.py

Claude AI-powered citation type detection for ambiguous inputs.
Drop-in replacement for gemini_router.py with identical interface.
"""

import re
import json
import os
from typing import Optional, Tuple

import anthropic

from models import CitationType, CitationMetadata
from config import DEFAULT_TIMEOUT


class ClaudeRouter:
    """Uses Claude to classify ambiguous citation queries."""
    
    SYSTEM_PROMPT = """You are a citation classifier. Analyze the input and determine what type of source it references.

Even if the input is fragmentary, incomplete, or contains only partial information (like author names, partial titles, or abbreviated references), make your best inference about the citation type.

Classify as one of:
- journal: Academic journal articles, peer-reviewed papers
- book: Books, book chapters, monographs
- legal: Court cases, statutes, legal documents
- interview: Interviews, oral histories, personal communications
- newspaper: Newspaper articles, magazine articles, news reports
- government: Government reports, congressional records, official documents
- medical: Medical records, clinical studies (if clearly medical context)
- archival: Depositions, unpublished manuscripts, archival materials
- url: Websites, online resources
- unknown: Cannot determine with any confidence

Respond in JSON only, no other text:
{"type": "...", "confidence": 0.0-1.0, "title": "", "authors": [], "year": "", "reasoning": "brief explanation"}"""

    def __init__(self, api_key: Optional[str] = None, timeout: int = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def classify(self, text: str) -> Tuple[CitationType, Optional[CitationMetadata]]:
        if not self.client:
            return CitationType.UNKNOWN, None
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Classify this citation:\n\n{text}"}]
            )
            
            response_text = response.content[0].text
            return self._parse_response(response_text, text)
            
        except anthropic.RateLimitError:
            print("[ClaudeRouter] Rate limited")
            return CitationType.UNKNOWN, None
        except anthropic.AuthenticationError:
            print("[ClaudeRouter] Authentication failed - check ANTHROPIC_API_KEY")
            return CitationType.UNKNOWN, None
        except Exception as e:
            print(f"[ClaudeRouter] Error: {e}")
            return CitationType.UNKNOWN, None
    
    def _parse_response(self, response_text: str, original: str) -> Tuple[CitationType, Optional[CitationMetadata]]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return CitationType.UNKNOWN, None
            
            data = json.loads(json_match.group())
            
            type_map = {
                'journal': CitationType.JOURNAL,
                'book': CitationType.BOOK,
                'legal': CitationType.LEGAL,
                'interview': CitationType.INTERVIEW,
                'newspaper': CitationType.NEWSPAPER,
                'government': CitationType.GOVERNMENT,
                'medical': CitationType.MEDICAL,
                'archival': CitationType.ARCHIVAL if hasattr(CitationType, 'ARCHIVAL') else CitationType.UNKNOWN,
                'url': CitationType.URL,
            }
            
            citation_type = type_map.get(data.get('type', '').lower(), CitationType.UNKNOWN)
            
            if citation_type == CitationType.UNKNOWN:
                return citation_type, None
            
            metadata = CitationMetadata(
                citation_type=citation_type,
                raw_source=original,
                source_engine="Claude Router",
                title=data.get('title', ''),
                authors=data.get('authors', []),
                year=data.get('year'),
                confidence=data.get('confidence', 0.5),
                notes=data.get('reasoning', ''),
            )
            
            return citation_type, metadata
            
        except json.JSONDecodeError:
            return CitationType.UNKNOWN, None
        except Exception:
            return CitationType.UNKNOWN, None


def classify_with_claude(text: str) -> Tuple[CitationType, Optional[CitationMetadata]]:
    """Convenience function matching classify_with_gemini interface."""
    return ClaudeRouter().classify(text)
