"""
citeflex/unified_router.py

Unified routing logic combining the best of CiteFlex Pro and Cite Fix Pro.

Version History:
    2025-12-05 13:15 V1.0: Initial unified router combining both systems
    2025-12-05 13:15 V1.1: Added Westlaw pattern, verified all medical .gov exclusions

KEY IMPROVEMENTS OVER ORIGINAL router.py:
1. Legal detection uses court.is_legal_citation() which checks FAMOUS_CASES cache
   during detection (not just regex patterns that miss bare case names)
2. Legal extraction uses court.extract_metadata() for cache + CourtListener API
3. Book search uses books.py's GoogleBooksAPI + OpenLibraryAPI with PUBLISHER_PLACE_MAP
4. Academic search uses CiteFlex Pro's parallel engine execution
5. Medical URL override prevents PubMed/NIH URLs from routing to government

ARCHITECTURE:
- Wrapper classes convert court.py/books.py dicts → CitationMetadata
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

# Import Cite Fix Pro modules (at root level)
import court
import books


# =============================================================================
# CONFIGURATION
# =============================================================================

PARALLEL_TIMEOUT = 6  # seconds (reduced from 12 for faster response)
MAX_WORKERS = 4

# Medical domains that should NOT route to government engine
MEDICAL_DOMAINS = ['pubmed', 'ncbi.nlm.nih.gov', 'nih.gov/health', 'medlineplus']


# =============================================================================
# ENGINE INSTANCES (reused across requests)
# =============================================================================

_crossref = CrossrefEngine()
_openalex = OpenAlexEngine()
_semantic = SemanticScholarEngine()


# =============================================================================
# RESULTS CACHE (speeds up repeated queries)
# =============================================================================

import time
import threading
import hashlib

class ResultsCache:
    """
    Thread-safe cache for citation search results.
    Speeds up repeated queries significantly.
    
    Added: 2025-12-05 15:30
    """
    
    TTL_SECONDS = 1800  # 30 minutes
    MAX_SIZE = 200      # Maximum cached queries
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
    
    def _make_key(self, query: str, style: str) -> str:
        """Create cache key from query + style."""
        normalized = f"{query.lower().strip()}|{style}"
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, query: str, style: str):
        """Get cached results if not expired."""
        key = self._make_key(query, style)
        with self._lock:
            if key in self._cache:
                results, timestamp = self._cache[key]
                if time.time() - timestamp < self.TTL_SECONDS:
                    print(f"[Cache] HIT for: {query[:30]}...")
                    return results
                else:
                    del self._cache[key]  # Expired
        return None
    
    def set(self, query: str, style: str, results):
        """Cache results with timestamp."""
        key = self._make_key(query, style)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.MAX_SIZE:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[key] = (results, time.time())
            print(f"[Cache] STORED: {query[:30]}... ({len(self._cache)} cached)")
    
    def clear(self):
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()

# Global cache instance
_results_cache = ResultsCache()
_pubmed = PubMedEngine()


# =============================================================================
# WRAPPER: CONVERT COURT.PY DICT → CitationMetadata
# =============================================================================

def _legal_dict_to_metadata(data: dict, raw_source: str) -> Optional[CitationMetadata]:
    """Convert court.py extract_metadata() dict to CitationMetadata."""
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
# UNIFIED LEGAL SEARCH (uses court.py)
# =============================================================================

def _route_legal(query: str) -> Optional[CitationMetadata]:
    """
    Route legal case queries using Cite Fix Pro's court.py.
    
    This is superior to CiteFlex Pro's legal.py because:
    1. FAMOUS_CASES cache has 100+ landmark cases
    2. is_legal_citation() checks cache during detection (catches "Roe v Wade")
    3. Fuzzy matching via difflib for near-matches
    4. CourtListener API fallback with phrase/keyword/fuzzy attempts
    """
    try:
        data = court.extract_metadata(query)
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
    
    # Fallback to CiteFlex Pro's Crossref (has book chapters)
    try:
        result = _crossref.search(query)
        if result and result.has_minimum_data():
            return result
    except Exception:
        pass
    
    return None


# =============================================================================
# PARALLEL ENGINE EXECUTION (from CiteFlex Pro)
# =============================================================================

def _search_engines_parallel(
    engines: List[Tuple[str, callable]],
    query: str,
    timeout: float = PARALLEL_TIMEOUT
) -> Optional[CitationMetadata]:
    """
    Execute multiple search engines in parallel, return first valid result.
    
    This reduces worst-case latency from 40+ seconds to ~10 seconds.
    """
    if not engines:
        return None
    
    with ThreadPoolExecutor(max_workers=min(len(engines), MAX_WORKERS)) as executor:
        future_to_engine = {
            executor.submit(fn, query): name
            for name, fn in engines
        }
        
        try:
            for future in as_completed(future_to_engine, timeout=timeout):
                engine_name = future_to_engine[future]
                try:
                    result = future.result(timeout=1)
                    if result and result.has_minimum_data():
                        print(f"[UnifiedRouter] Found via {engine_name}")
                        return result
                except Exception as e:
                    print(f"[UnifiedRouter] {engine_name} failed: {e}")
                    continue
        except FuturesTimeout:
            print(f"[UnifiedRouter] Parallel search timed out after {timeout}s")
    
    return None


# =============================================================================
# JOURNAL/ACADEMIC ROUTING (CiteFlex Pro engines)
# =============================================================================

def _route_journal(query: str) -> Optional[CitationMetadata]:
    """
    Route journal/article queries using CiteFlex Pro's academic engines.
    Runs Crossref, OpenAlex, and Semantic Scholar in parallel.
    """
    # Check for DOI in query first (instant lookup)
    doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', query)
    if doi_match:
        doi = doi_match.group(1).rstrip('.,;')
        result = _crossref.get_by_id(doi)
        if result:
            print("[UnifiedRouter] Found via direct DOI lookup")
            return result
    
    # Parallel search across academic engines
    engines = [
        ("Crossref", _crossref.search),
        ("OpenAlex", _openalex.search),
        ("Semantic Scholar", _semantic.search),
    ]
    
    return _search_engines_parallel(engines, query)


# =============================================================================
# MEDICAL ROUTING (CiteFlex Pro PubMed + academic engines)
# =============================================================================

def _route_medical(query: str) -> Optional[CitationMetadata]:
    """
    Route medical/clinical queries.
    Tries PubMed first (specialized), then falls back to academic engines.
    """
    # Check for PMID
    pmid_match = re.search(r'(?:pmid:?\s*|pubmed:?\s*)(\d+)', query, re.IGNORECASE)
    if pmid_match:
        pmid = pmid_match.group(1)
        result = _pubmed.get_by_id(pmid)
        if result:
            print("[UnifiedRouter] Found via direct PMID lookup")
            return result
    
    # Parallel search: PubMed + academic engines
    engines = [
        ("PubMed", _pubmed.search),
        ("Crossref", _crossref.search),
        ("Semantic Scholar", _semantic.search),
    ]
    
    return _search_engines_parallel(engines, query)


# =============================================================================
# URL ROUTING (with medical domain override)
# =============================================================================

def _is_medical_url(url: str) -> bool:
    """Check if URL is from a medical domain (should route to PubMed, not gov)."""
    lower = url.lower()
    return any(domain in lower for domain in MEDICAL_DOMAINS)


def _is_newspaper_url(url: str) -> bool:
    """Check if URL is from a newspaper domain."""
    lower = url.lower()
    return any(domain in lower for domain in NEWSPAPER_DOMAINS.keys())


def _is_government_url(url: str) -> bool:
    """Check if URL is from a government domain."""
    return '.gov' in url.lower() and not _is_medical_url(url)


def _route_url(url: str) -> Optional[CitationMetadata]:
    """
    Route URL-based queries with smart domain detection.
    
    Priority:
    1. Medical URLs → PubMed (override .gov for NIH/PubMed)
    2. Academic publisher URLs → DOI extraction → Crossref
    3. Government URLs → basic metadata extraction
    4. Newspaper URLs → basic metadata extraction
    5. Generic URLs → basic metadata
    """
    # 1. Medical URL override (PubMed, NIH, etc.)
    if _is_medical_url(url):
        # Try to extract PMID from URL
        pmid_match = re.search(r'/(\d{7,8})/?', url)
        if pmid_match:
            result = _pubmed.get_by_id(pmid_match.group(1))
            if result:
                result.url = url
                return result
        # Fall back to medical routing
        return _route_medical(url)
    
    # 2. Academic publisher URL → DOI extraction
    if is_academic_publisher_url(url):
        doi = extract_doi_from_url(url)
        if doi:
            result = _crossref.get_by_id(doi)
            if result:
                result.url = url
                print("[UnifiedRouter] Found via DOI extraction from URL")
                return result
    
    # 3. Government URL
    if _is_government_url(url):
        return extract_by_type(url, CitationType.GOVERNMENT)
    
    # 4. Newspaper URL
    if _is_newspaper_url(url):
        return extract_by_type(url, CitationType.NEWSPAPER)
    
    # 5. Generic URL
    return extract_by_type(url, CitationType.URL)


# =============================================================================
# MAIN ROUTING FUNCTION
# =============================================================================

def route_citation(query: str) -> Tuple[Optional[CitationMetadata], DetectionResult]:
    """
    Main routing function with unified detection and search.
    
    KEY DIFFERENCE from original router.py:
    - Uses court.is_legal_citation() for legal detection (cache-aware)
    - This catches bare case names like "Roe v Wade" that regex misses
    """
    query = query.strip()
    
    # ==========================================================================
    # STEP 1: Check if it's a legal citation using court.py's cache-aware detector
    # ==========================================================================
    if court.is_legal_citation(query):
        print(f"[UnifiedRouter] Detected: LEGAL (cache-aware)")
        metadata = _route_legal(query)
        if metadata:
            return metadata, DetectionResult(
                citation_type=CitationType.LEGAL,
                confidence=0.95,
                cleaned_query=query
            )
    
    # ==========================================================================
    # STEP 2: Use CiteFlex Pro's pattern detection for other types
    # ==========================================================================
    detection = detect_type(query)
    print(f"[UnifiedRouter] Detected: {detection.citation_type.name} (confidence: {detection.confidence})")
    
    metadata = None
    
    # ==========================================================================
    # STEP 3: Route based on detected type
    # ==========================================================================
    
    if detection.citation_type == CitationType.INTERVIEW:
        metadata = extract_by_type(query, CitationType.INTERVIEW)
    
    elif detection.citation_type == CitationType.LEGAL:
        # Already checked above, but try again in case detection differs
        metadata = _route_legal(query)
    
    elif detection.citation_type == CitationType.GOVERNMENT:
        metadata = extract_by_type(query, CitationType.GOVERNMENT)
    
    elif detection.citation_type == CitationType.NEWSPAPER:
        metadata = extract_by_type(query, CitationType.NEWSPAPER)
    
    elif detection.citation_type == CitationType.MEDICAL:
        metadata = _route_medical(query)
    
    elif detection.citation_type == CitationType.JOURNAL:
        metadata = _route_journal(query)
    
    elif detection.citation_type == CitationType.BOOK:
        metadata = _route_book(query)
    
    elif detection.citation_type == CitationType.URL:
        metadata = _route_url(query)
    
    elif detection.citation_type == CitationType.UNKNOWN:
        # Try journal engines as default, then book
        metadata = _route_journal(query)
        if not metadata:
            metadata = _route_book(query)
    
    return metadata, detection


# =============================================================================
# HIGH-LEVEL API (same interface as original router.py)
# =============================================================================

def get_citation(
    query: str,
    style: str = "Chicago Manual of Style"
) -> Tuple[Optional[CitationMetadata], Optional[str]]:
    """
    Main entry point for getting a formatted citation.
    
    This is the function imported by document_processor.py.
    
    Args:
        query: The citation query (URL, title, case name, etc.)
        style: Citation style name
        
    Returns:
        Tuple of (CitationMetadata, formatted_citation_string)
        Both may be None if lookup fails.
    """
    # Check cache first (use "single:" prefix to distinguish from multiple)
    cache_key = f"single:{query}"
    cached = _results_cache.get(cache_key, style)
    if cached is not None:
        return cached
    
    metadata, detection = route_citation(query)
    
    # FALLBACK: If no results, try Semantic Scholar directly (best for messy queries)
    if not metadata or not metadata.has_minimum_data():
        print(f"[UnifiedRouter] Primary routing failed, trying Semantic Scholar fallback...")
        try:
            metadata = _semantic.search(query)
            if not metadata:
                # Try with cleaned query (remove noise words)
                words = [w for w in query.split() if len(w) > 3 and w.lower() not in 
                         {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'none', 'could', 'would', 'put'}]
                if words:
                    clean_query = ' '.join(words[:5])
                    metadata = _semantic.search(clean_query)
        except Exception as e:
            print(f"[UnifiedRouter] Semantic fallback failed: {e}")
    
    if not metadata or not metadata.has_minimum_data():
        print(f"[UnifiedRouter] No metadata found for: {query[:50]}...")
        return None, None
    
    formatter = get_formatter(style)
    formatted = formatter.format(metadata)
    
    # Cache the result
    _results_cache.set(cache_key, style, (metadata, formatted))
    
    return metadata, formatted


def get_multiple_citations(
    query: str,
    style: str = "Chicago Manual of Style",
    limit: int = 5
) -> List[Tuple[CitationMetadata, str, str]]:
    """
    Get multiple citation options from ALL relevant engines in parallel.
    
    Args:
        query: Search query
        style: Citation style
        limit: Maximum results
        
    Returns:
        List of (CitationMetadata, formatted_string, source_name) tuples
    """
    # Check cache first (instant return if hit)
    cached = _results_cache.get(query, style)
    if cached is not None:
        return cached[:limit]
    
    results = []
    formatter = get_formatter(style)
    seen_titles = set()  # Deduplicate by title
    
    def add_result(meta, source_name):
        """Add result if valid and not duplicate."""
        if meta and meta.has_minimum_data():
            title_key = (meta.title or '').lower()[:50]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                formatted = formatter.format(meta)
                results.append((meta, formatted, source_name))
    
    # 1. Always check legal cache first (instant)
    if court.is_legal_citation(query):
        metadata = _route_legal(query)
        if metadata:
            add_result(metadata, "Legal Cache")
    
    # 2. Query ALL engines in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def search_crossref():
        try:
            metas = _crossref.search_multiple(query, 2)
            return [("Crossref", m) for m in metas if m]
        except: return []
    
    def search_openalex():
        try:
            meta = _openalex.search(query)
            return [("OpenAlex", meta)] if meta else []
        except: return []
    
    def search_semantic():
        try:
            meta = _semantic.search(query)
            return [("Semantic Scholar", meta)] if meta else []
        except: return []
    
    def search_pubmed():
        try:
            meta = _pubmed.search(query)
            return [("PubMed", meta)] if meta else []
        except: return []
    
    def search_books():
        try:
            print(f"[search_books] Searching all book engines for: {query[:50]}...")
            # Use search_all_engines to get results from ALL book APIs
            book_results = books.search_all_engines(query)
            print(f"[search_books] Got {len(book_results) if book_results else 0} results from all engines")
            results = []
            for data in book_results[:4]:  # Up to 4 book results
                meta = _book_dict_to_metadata(data, query)
                if meta:
                    # Use source_engine from books.py (Google Books, LOC, WorldCat, Open Library)
                    source = data.get('source_engine', 'Books')
                    results.append((source, meta))
                    print(f"[search_books] Found: {meta.title[:50] if meta.title else 'No title'}... [{source}]")
            return results
        except Exception as e:
            print(f"[search_books] Error: {e}")
            return []
    
    # Run all searches in parallel
    engine_searches = [
        search_crossref,
        search_openalex, 
        search_semantic,
        search_pubmed,
        search_books,
    ]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fn) for fn in engine_searches]
        
        for future in as_completed(futures, timeout=10):
            try:
                # Increased timeout to 3s per result (was 1s, cutting off Semantic Scholar)
                engine_results = future.result(timeout=3)
                for source_name, meta in engine_results:
                    add_result(meta, source_name)
            except Exception as e:
                continue
    
    # FALLBACK: For messy queries, try Semantic Scholar with relaxed search if few results
    if len(results) < 2:
        print(f"[UnifiedRouter] Few results ({len(results)}), trying Semantic Scholar fallback...")
        try:
            # Try with just key words (strip common words)
            words = [w for w in query.split() if len(w) > 3 and w.lower() not in 
                     {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'none', 'could', 'would'}]
            if words:
                fallback_query = ' '.join(words[:4])  # First 4 significant words
                meta = _semantic.search(fallback_query)
                if meta:
                    add_result(meta, "Semantic Scholar (fuzzy)")
        except Exception as e:
            print(f"[UnifiedRouter] Semantic fallback failed: {e}")
    
    # Cache results for future queries
    if results:
        _results_cache.set(query, style, results)
    
    return results[:limit]


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def search_citation(query: str) -> List[dict]:
    """
    Backward-compatible search function.
    Returns list of dicts (matching old search.py interface).
    """
    results = []
    
    # Try legal first
    if court.is_legal_citation(query):
        data = court.extract_metadata(query)
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
