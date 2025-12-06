"""
citeflex/unified_router.py

Unified routing logic combining the best of CiteFlex Pro and Cite Fix Pro.

Version History:
    2025-12-06 11:50 V3.1: CRITICAL FIX - Changed search_by_doi to get_by_id,
                           added famous papers cache to _route_journal,
                           added fallback DOI regex extraction to _route_url
    2025-12-06 V3.0: Switched to Claude API as primary AI router
    2025-12-05 13:15 V1.0: Initial unified router combining both systems
    2025-12-05 13:15 V1.1: Added Westlaw pattern, verified all medical .gov exclusions
    2025-12-05 20:30 V2.0: Moved to engines/ architecture (superlegal, books)
    2025-12-05 21:00 V2.1: Fixed get_multiple_citations to return 3-tuples
    2025-12-05 21:30 V2.2: Added URL/DOI handling to get_multiple_citations
    2025-12-05 22:30 V2.3: Added famous papers cache (10,000 most-cited papers)
    2025-12-05 23:00 V2.4: Added Gemini AI fallback for UNKNOWN queries
    2025-12-05 22:45 V2.4: Fixed UNKNOWN routing to search books first
    2025-12-06 V3.0: Switched to Claude API as primary AI router

KEY IMPROVEMENTS OVER ORIGINAL router.py:
1. Legal detection uses superlegal.is_legal_citation() which checks FAMOUS_CASES cache
   during detection (not just regex patterns that miss bare case names)
2. Legal extraction uses superlegal.extract_metadata() for cache + CourtListener API
3. Book search uses books.py's GoogleBooksAPI + OpenLibraryAPI with PUBLISHER_PLACE_MAP
4. Academic search uses CiteFlex Pro's parallel engine execution
5. Medical URL override prevents PubMed/NIH URLs from routing to government
6. Claude AI router for ambiguous queries with multi-option support

ARCHITECTURE:
- Wrapper classes convert superlegal.py/books.py dicts → CitationMetadata
- Parallel execution via ThreadPoolExecutor (12s timeout)
- Routing priority: Legal → URL handling → Parallel search → Fallback
"""

import re
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

from models import CitationMetadata, CitationType
from config import NEWSPAPER_DOMAINS, GOV_AGENCY_MAP
from detectors import detect_type, DetectionResult, is_url
from extractors import extract_by_type
from formatters.base import get_formatter

# Import CiteFlex Pro engines
from engines.academic import CrossrefEngine, OpenAlexEngine, SemanticScholarEngine, PubMedEngine
from engines.doi import extract_doi_from_url, is_academic_publisher_url

# Import Cite Fix Pro modules (now in engines/)
from engines import superlegal
from engines import books
from engines.famous_papers import find_famous_paper

# =============================================================================
# AI ROUTER CONFIGURATION (Claude primary, Gemini fallback)
# =============================================================================

import os
AI_ROUTER = os.environ.get('AI_ROUTER', 'claude').lower()  # 'claude' or 'gemini'

# Try to import Claude router (primary)
try:
    from claude_router import classify_with_claude, get_citation_options
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("[UnifiedRouter] Claude router not available")

# Try to import Gemini router (fallback)
try:
    from gemini_router import classify_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def classify_with_ai(query: str) -> Tuple[CitationType, Optional[CitationMetadata]]:
    """
    Use configured AI router (Claude preferred, Gemini fallback).
    Returns (CitationType, optional metadata).
    """
    if AI_ROUTER == 'claude' and CLAUDE_AVAILABLE:
        return classify_with_claude(query)
    elif AI_ROUTER == 'gemini' and GEMINI_AVAILABLE:
        return classify_with_gemini(query)
    elif CLAUDE_AVAILABLE:
        return classify_with_claude(query)
    elif GEMINI_AVAILABLE:
        return classify_with_gemini(query)
    else:
        return CitationType.UNKNOWN, None


AI_AVAILABLE = CLAUDE_AVAILABLE or GEMINI_AVAILABLE
if not AI_AVAILABLE:
    print("[UnifiedRouter] No AI router available - UNKNOWN queries will use default routing")
else:
    active_router = "CLAUDE" if (AI_ROUTER == 'claude' and CLAUDE_AVAILABLE) or (not GEMINI_AVAILABLE and CLAUDE_AVAILABLE) else "GEMINI"
    print(f"[UnifiedRouter] Using {active_router} for AI classification")


# =============================================================================
# CONFIGURATION
# =============================================================================

PARALLEL_TIMEOUT = 12  # seconds
MAX_WORKERS = 4

# Medical domains that should NOT route to government engine
MEDICAL_DOMAINS = ['pubmed', 'ncbi.nlm.nih.gov', 'nih.gov/health', 'medlineplus']


# =============================================================================
# ENGINE INSTANCES (reused across requests)
# =============================================================================

_crossref = CrossrefEngine()
_openalex = OpenAlexEngine()
_semantic = SemanticScholarEngine()
_pubmed = PubMedEngine()


# =============================================================================
# WRAPPER: CONVERT SUPERLEGAL.PY DICT → CitationMetadata
# =============================================================================

def _legal_dict_to_metadata(data: dict, raw_source: str) -> Optional[CitationMetadata]:
    """Convert superlegal.py extract_metadata() dict to CitationMetadata."""
    if not data:
        return None
    
    return CitationMetadata(
        citation_type=CitationType.LEGAL,
        raw_source=raw_source,
        source_engine=data.get('source_engine', 'Legal Cache/CourtListener'),
        case_name=data.get('case_name', ''),
        citation=data.get('citation', ''),
        court=data.get('court', ''),
        year=data.get('year', ''),
        jurisdiction=data.get('jurisdiction', 'US'),
        neutral_citation=data.get('neutral_citation', ''),
        url=data.get('url', ''),
        raw_data=data
    )


# =============================================================================
# WRAPPER: CONVERT BOOKS.PY DICT → CitationMetadata
# =============================================================================

def _book_dict_to_metadata(data: dict, raw_source: str) -> Optional[CitationMetadata]:
    """Convert books.py result dict to CitationMetadata."""
    if not data:
        return None
    
    return CitationMetadata(
        citation_type=CitationType.BOOK,
        raw_source=raw_source,
        source_engine=data.get('source_engine', 'Google Books/Open Library'),
        title=data.get('title', ''),
        authors=data.get('authors', []),
        year=data.get('year', ''),
        publisher=data.get('publisher', ''),
        place=data.get('place', ''),
        isbn=data.get('isbn', ''),
        raw_data=data
    )


# =============================================================================
# UNIFIED LEGAL SEARCH (uses superlegal.py)
# =============================================================================

def _route_legal(query: str) -> Optional[CitationMetadata]:
    """
    Route legal case queries using Cite Fix Pro's superlegal.py.
    
    This is superior to CiteFlex Pro's legal.py because:
    1. FAMOUS_CASES cache has 100+ landmark cases
    2. is_legal_citation() checks cache during detection (catches "Roe v Wade")
    3. Fuzzy matching via difflib for near-matches
    4. CourtListener API fallback with phrase/keyword/fuzzy attempts
    """
    try:
        data = superlegal.extract_metadata(query)
        if data and (data.get('case_name') or data.get('citation')):
            return _legal_dict_to_metadata(data, query)
    except Exception as e:
        print(f"[UnifiedRouter] Legal search error: {e}")
    
    return None


# =============================================================================
# UNIFIED BOOK SEARCH (uses books.py)
# =============================================================================

def _route_book(query: str) -> Optional[CitationMetadata]:
    """
    Route book queries using Cite Fix Pro's books.py.
    
    This is superior to CiteFlex Pro's google_cse.py because:
    1. Dual-engine: Open Library (precise ISBN) + Google Books (fuzzy search)
    2. PUBLISHER_PLACE_MAP fills in publication places
    3. ISBN detection routes to Open Library first
    """
    try:
        results = books.extract_metadata(query)
        if results and len(results) > 0:
            return _book_dict_to_metadata(results[0], query)
    except Exception as e:
        print(f"[UnifiedRouter] Book search error: {e}")
    
    return None


# =============================================================================
# UNIFIED JOURNAL SEARCH (parallel execution)
# =============================================================================

def _route_journal(query: str) -> Optional[CitationMetadata]:
    """
    Route journal/academic queries using parallel API execution.
    
    Engines tried (in parallel):
    1. Crossref - best for DOIs, formal citations
    2. OpenAlex - good coverage, fast
    3. Semantic Scholar - good for author+title queries
    4. PubMed - medical/life sciences
    """
    # Check famous papers cache first (instant lookup for 10,000 most-cited)
    famous = find_famous_paper(query)
    if famous:
        try:
            result = _crossref.get_by_id(famous["doi"])
            if result:
                print("[UnifiedRouter] Found via Famous Papers cache")
                return result
        except Exception:
            pass
    
    # Check for DOI in query (instant lookup)
    doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', query)
    if doi_match:
        doi = doi_match.group(1).rstrip('.,;')
        try:
            result = _crossref.get_by_id(doi)
            if result:
                print("[UnifiedRouter] Found via direct DOI lookup")
                return result
        except Exception:
            pass
    
    # Parallel search across academic engines
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_crossref.search, query): "Crossref",
            executor.submit(_openalex.search, query): "OpenAlex",
            executor.submit(_semantic.search, query): "Semantic Scholar",
            executor.submit(_pubmed.search, query): "PubMed",
        }
        
        for future in as_completed(futures, timeout=PARALLEL_TIMEOUT):
            engine_name = futures[future]
            try:
                result = future.result(timeout=2)
                if result and result.has_minimum_data():
                    result.source_engine = engine_name
                    results.append(result)
            except Exception:
                pass
    
    # Return best result (prefer one with DOI)
    if results:
        for r in results:
            if r.doi:
                return r
        return results[0]
    
    return None


# =============================================================================
# URL ROUTING
# =============================================================================

def _is_medical_url(url: str) -> bool:
    """Check if URL is a medical resource (PubMed, NIH, etc.)."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in MEDICAL_DOMAINS)


def _route_url(url: str) -> Optional[CitationMetadata]:
    """
    Route URL-based queries.
    
    Priority:
    1. Extract DOI from URL → Crossref lookup
    2. Academic publisher URL → Crossref search
    3. Medical URL → PubMed
    4. Newspaper URL → Newspaper extractor
    5. Government URL → Government extractor
    6. Generic URL → Basic metadata extraction
    """
    # Check for DOI in URL
    doi = extract_doi_from_url(url)
    if doi:
        try:
            result = _crossref.get_by_id(doi)
            if result and result.has_minimum_data():
                result.url = url
                return result
        except Exception:
            pass
    
    # Fallback: Try generic DOI extraction from URL path
    doi_match = re.search(r'(10\.\d{4,}/[^\s?#]+)', url)
    if doi_match:
        doi = doi_match.group(1).rstrip('.,;')
        try:
            result = _crossref.get_by_id(doi)
            if result and result.has_minimum_data():
                result.url = url
                print("[UnifiedRouter] Found via DOI in URL path")
                return result
        except Exception:
            pass
    
    # Check for academic publisher
    if is_academic_publisher_url(url):
        try:
            result = _crossref.search(url)
            if result and result.has_minimum_data():
                result.url = url
                return result
        except Exception:
            pass
    
    # Medical URLs go to PubMed
    if _is_medical_url(url):
        try:
            result = _pubmed.search(url)
            if result and result.has_minimum_data():
                result.url = url
                return result
        except Exception:
            pass
    
    # Fallback to standard extraction
    return extract_by_type(url, CitationType.URL)


# =============================================================================
# MAIN ROUTING FUNCTION
# =============================================================================

def route_citation(query: str, style: str = "chicago") -> Tuple[Optional[CitationMetadata], str]:
    """
    Main entry point: route query to appropriate engine and format result.
    
    Returns: (CitationMetadata, formatted_citation_string)
    """
    query = query.strip()
    if not query:
        return None, ""
    
    formatter = get_formatter(style)
    metadata = None
    
    # 1. Check for legal citation FIRST (superlegal.py handles famous cases)
    if superlegal.is_legal_citation(query):
        metadata = _route_legal(query)
        if metadata:
            return metadata, formatter.format(metadata)
    
    # 2. Check for URL
    if is_url(query):
        metadata = _route_url(query)
        if metadata:
            return metadata, formatter.format(metadata)
    
    # 3. Detect type using standard detectors
    detection = detect_type(query)
    
    # 4. Route based on detection
    if detection.citation_type == CitationType.LEGAL:
        metadata = _route_legal(query)
    
    elif detection.citation_type == CitationType.BOOK:
        metadata = _route_book(query)
    
    elif detection.citation_type in [CitationType.JOURNAL, CitationType.MEDICAL]:
        # Check famous papers cache first
        famous = find_famous_paper(query)
        if famous:
            metadata = CitationMetadata(
                citation_type=CitationType.JOURNAL,
                raw_source=query,
                source_engine="Famous Papers Cache",
                **famous
            )
        else:
            metadata = _route_journal(query)
    
    elif detection.citation_type == CitationType.NEWSPAPER:
        metadata = extract_by_type(query, CitationType.NEWSPAPER)
    
    elif detection.citation_type == CitationType.GOVERNMENT:
        metadata = extract_by_type(query, CitationType.GOVERNMENT)
    
    elif detection.citation_type == CitationType.INTERVIEW:
        metadata = extract_by_type(query, CitationType.INTERVIEW)
    
    else:
        # UNKNOWN: Try AI classification first
        if AI_AVAILABLE:
            ai_type, ai_meta = classify_with_ai(query)
            if ai_type != CitationType.UNKNOWN:
                print(f"[UnifiedRouter] AI classified as: {ai_type.name}")
                
                if ai_type == CitationType.BOOK:
                    metadata = _route_book(query)
                elif ai_type == CitationType.LEGAL:
                    metadata = _route_legal(query)
                elif ai_type in [CitationType.JOURNAL, CitationType.MEDICAL]:
                    metadata = _route_journal(query)
                elif ai_type == CitationType.NEWSPAPER:
                    metadata = extract_by_type(query, CitationType.NEWSPAPER)
                elif ai_type == CitationType.GOVERNMENT:
                    metadata = extract_by_type(query, CitationType.GOVERNMENT)
        
        # Fallback: try books first, then journals
        if not metadata:
            metadata = _route_book(query)
        if not metadata:
            metadata = _route_journal(query)
    
    # Format and return
    if metadata:
        return metadata, formatter.format(metadata)
    
    return None, ""


# =============================================================================
# MULTIPLE RESULTS FUNCTION
# =============================================================================

def get_multiple_citations(query: str, style: str = "chicago", limit: int = 5) -> List[Tuple[CitationMetadata, str, str]]:
    """
    Get multiple citation candidates for user selection.
    
    Returns list of (metadata, formatted_citation, source_name) tuples.
    """
    query = query.strip()
    if not query:
        return []
    
    formatter = get_formatter(style)
    results = []
    
    # Detect type
    detection = detect_type(query)
    
    # Check for URL with DOI first
    if is_url(query):
        doi = extract_doi_from_url(query)
        if doi:
            try:
                result = _crossref.get_by_id(doi)
                if result and result.has_minimum_data():
                    result.url = query
                    formatted = formatter.format(result)
                    results.append((result, formatted, "Crossref (DOI)"))
            except Exception:
                pass
    
    # Check for legal citation
    if superlegal.is_legal_citation(query) or detection.citation_type == CitationType.LEGAL:
        metadata = _route_legal(query)
        if metadata:
            formatted = formatter.format(metadata)
            results.append((metadata, formatted, "Legal Cache"))
        return results  # Legal citations typically have one authoritative result
    
    # For journals/academic
    if detection.citation_type in [CitationType.JOURNAL, CitationType.MEDICAL, CitationType.UNKNOWN]:
        # Check famous papers first
        famous = find_famous_paper(query)
        if famous:
            meta = CitationMetadata(
                citation_type=CitationType.JOURNAL,
                raw_source=query,
                source_engine="Famous Papers Cache",
                **famous
            )
            formatted = formatter.format(meta)
            results.append((meta, formatted, "Famous Papers"))
        
        # Query multiple engines
        try:
            metadatas = _crossref.search_multiple(query, limit)
            for meta in metadatas:
                if meta and meta.has_minimum_data():
                    formatted = formatter.format(meta)
                    results.append((meta, formatted, "Crossref"))
        except Exception:
            pass
        
        # Add Semantic Scholar results
        if len(results) < limit:
            try:
                ss_result = _semantic.search(query)
                if ss_result and ss_result.has_minimum_data():
                    is_duplicate = any(
                        ss_result.title and r[0].title and 
                        ss_result.title.lower()[:30] == r[0].title.lower()[:30]
                        for r in results
                    )
                    if not is_duplicate:
                        formatted = formatter.format(ss_result)
                        results.append((ss_result, formatted, "Semantic Scholar"))
            except Exception:
                pass
    
    elif detection.citation_type == CitationType.BOOK:
        # Query book engines
        try:
            book_results = books.extract_metadata(query)
            for data in book_results[:limit]:
                meta = _book_dict_to_metadata(data, query)
                if meta:
                    formatted = formatter.format(meta)
                    source = meta.source_engine or data.get('source_engine', 'Google Books')
                    results.append((meta, formatted, source))
        except Exception:
            pass
        
        # Also try Crossref (has book chapters)
        if len(results) < limit:
            try:
                metadatas = _crossref.search_multiple(query, limit - len(results))
                for meta in metadatas:
                    if meta and meta.has_minimum_data():
                        formatted = formatter.format(meta)
                        results.append((meta, formatted, "Crossref"))
            except Exception:
                pass
        
        # Also try Semantic Scholar
        if len(results) < limit:
            try:
                ss_result = _semantic.search(query)
                if ss_result and ss_result.has_minimum_data():
                    is_duplicate = any(
                        ss_result.title and r[0].title and 
                        ss_result.title.lower()[:30] == r[0].title.lower()[:30]
                        for r in results
                    )
                    if not is_duplicate:
                        formatted = formatter.format(ss_result)
                        results.append((ss_result, formatted, "Semantic Scholar"))
            except Exception:
                pass
    
    elif detection.citation_type == CitationType.UNKNOWN:
        # Try AI router to classify ambiguous queries
        if AI_AVAILABLE:
            ai_type, ai_meta = classify_with_ai(query)
            if ai_type != CitationType.UNKNOWN:
                print(f"[UnifiedRouter] AI classified as: {ai_type.name}")
                
                # Route based on AI's classification
                if ai_type == CitationType.BOOK:
                    try:
                        book_results = books.extract_metadata(query)
                        for data in book_results[:limit]:
                            meta = _book_dict_to_metadata(data, query)
                            if meta:
                                formatted = formatter.format(meta)
                                source = data.get('source_engine', 'Google Books')
                                results.append((meta, formatted, source))
                    except Exception:
                        pass
                    # Also try Semantic Scholar
                    if len(results) < limit:
                        try:
                            ss_result = _semantic.search(query)
                            if ss_result and ss_result.has_minimum_data():
                                is_duplicate = any(
                                    ss_result.title and r[0].title and 
                                    ss_result.title.lower()[:30] == r[0].title.lower()[:30]
                                    for r in results
                                )
                                if not is_duplicate:
                                    formatted = formatter.format(ss_result)
                                    results.append((ss_result, formatted, "Semantic Scholar"))
                        except Exception:
                            pass
                    return results[:limit]
                
                elif ai_type == CitationType.LEGAL:
                    metadata = _route_legal(query)
                    if metadata:
                        formatted = formatter.format(metadata)
                        results.append((metadata, formatted, "Legal Cache"))
                    return results
                
                elif ai_type in [CitationType.JOURNAL, CitationType.MEDICAL]:
                    metadatas = _crossref.search_multiple(query, limit)
                    for meta in metadatas:
                        if meta and meta.has_minimum_data():
                            formatted = formatter.format(meta)
                            results.append((meta, formatted, "Crossref"))
                    # Also try Semantic Scholar
                    if len(results) < limit:
                        try:
                            ss_result = _semantic.search(query)
                            if ss_result and ss_result.has_minimum_data():
                                is_duplicate = any(
                                    ss_result.title and r[0].title and 
                                    ss_result.title.lower()[:30] == r[0].title.lower()[:30]
                                    for r in results
                                )
                                if not is_duplicate:
                                    formatted = formatter.format(ss_result)
                                    results.append((ss_result, formatted, "Semantic Scholar"))
                        except Exception:
                            pass
                    return results[:limit]
        
        # Fallback: try book engines FIRST (often what users want)
        try:
            book_results = books.extract_metadata(query)
            for data in book_results[:3]:
                meta = _book_dict_to_metadata(data, query)
                if meta:
                    formatted = formatter.format(meta)
                    source = data.get('source_engine', 'Google Books')
                    results.append((meta, formatted, source))
        except Exception:
            pass
        
        # Then fill remaining with Crossref (journals, chapters)
        if len(results) < limit:
            try:
                metadatas = _crossref.search_multiple(query, limit - len(results))
                for meta in metadatas:
                    if meta and meta.has_minimum_data():
                        formatted = formatter.format(meta)
                        results.append((meta, formatted, "Crossref"))
            except Exception:
                pass
        
        # Finally try Semantic Scholar
        if len(results) < limit:
            try:
                ss_result = _semantic.search(query)
                if ss_result and ss_result.has_minimum_data():
                    is_duplicate = any(
                        ss_result.title and r[0].title and 
                        ss_result.title.lower()[:30] == r[0].title.lower()[:30]
                        for r in results
                    )
                    if not is_duplicate:
                        formatted = formatter.format(ss_result)
                        results.append((ss_result, formatted, "Semantic Scholar"))
            except Exception:
                pass
    
    return results[:limit]


# =============================================================================
# MULTI-OPTION CITATIONS (uses Claude's get_citation_options)
# =============================================================================

def get_citation_options_formatted(query: str, style: str = "chicago", limit: int = 5) -> List[dict]:
    """
    Get multiple citation options using Claude AI + multiple APIs.
    
    This is the preferred method for ambiguous queries like "Caplan mind games".
    Returns list of dicts with {citation, source, title, authors, year, ...}.
    
    Uses claude_router.get_citation_options() which searches:
    - Google Books
    - Crossref  
    - PubMed
    - Famous Cases Cache
    """
    if CLAUDE_AVAILABLE:
        try:
            return get_citation_options(query, max_options=limit)
        except Exception as e:
            print(f"[UnifiedRouter] Claude options error: {e}")
    
    # Fallback to standard multiple citations
    results = get_multiple_citations(query, style, limit)
    return [
        {
            "citation": formatted,
            "source": source,
            "title": meta.title if meta else "",
            "authors": meta.authors if meta else [],
            "year": meta.year if meta else ""
        }
        for meta, formatted, source in results
    ]


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Alias for app.py compatibility
def get_citation(query: str, style: str = "chicago") -> Tuple[Optional[CitationMetadata], str]:
    """Alias for route_citation() - backward compatibility."""
    return route_citation(query, style)


def search_citation(query: str) -> List[dict]:
    """
    Backward-compatible search function.
    Returns list of dicts (matching old search.py interface).
    """
    results = []
    
    # Try legal first
    if superlegal.is_legal_citation(query):
        data = superlegal.extract_metadata(query)
        if data:
            results.append(data)
        return results
    
    # Try books
    try:
        book_results = books.extract_metadata(query)
        results.extend(book_results)
    except Exception:
        pass
    
    # Try academic
    try:
        meta = _route_journal(query)
        if meta:
            results.append(meta.to_dict())
    except Exception:
        pass
    
    return results
