import pandas as pd
import re
import html
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Data loader for tourism-related CSV files with robust handling and metadata extraction.
class TourismDataLoader:
   
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.documents: List[Dict[str, Any]] = []

    # Load all CSV files and extract documents with metadata.    
    def load_all(self) -> List[Dict[str, Any]]:
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        csv_files = sorted(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        all_docs = []
        for csv_file in csv_files:
            try:
                docs = self._load_file(csv_file)
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {csv_file.name}")
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {e}")
                continue
        
        self.documents = all_docs
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    # Load a single CSV file and extract documents with metadata.
    def _load_file(self, filepath: Path) -> List[Dict[str, Any]]:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Could not decode {filepath} with any encoding")
        
        # Clean column names
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        
        documents = []
        for idx, row in df.iterrows():
            try:
                doc = self._process_row(row, filepath.stem, idx)
                if doc and doc.get('content'):
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing row {idx} in {filepath.name}: {e}")
                continue
        
        return documents
    
    # Process a single row to extract content and metadata, with robust handling of various formats and fallbacks.
    def _process_row(self, row: pd.Series, source_file: str, row_idx: int) -> Optional[Dict[str, Any]]:
        # Extract content from various possible column names
        content = self._extract_content(row)
        if not content or len(content) < 20:  # Skip very short entries
            return None
        
        # Clean content
        content = self._clean_text(content)
        
        # Extract metadata with fallbacks
        metadata = {
            'source_file': source_file,
            'source_row': row_idx,
            'title': self._get_column_value(row, ['title', 'name', 'attraction_name', 'hotel_name', 'place_name']),
            'location': self._get_column_value(row, ['location', 'destination', 'place', 'region', 'city', 'area']),
            'category': self._get_column_value(row, ['category', 'type', 'attraction_type', 'place_type']) or 'general',
            'price_range': self._get_column_value(row, ['price_range', 'price', 'budget', 'cost', 'price_category']),
            'best_season': self._get_column_value(row, ['best_season', 'season', 'best_time', 'recommended_time']),
            'activities': self._parse_list_field(row, ['activities', 'things_to_do', 'available_activities']),
            'url': self._get_column_value(row, ['url', 'link', 'website', 'source_url']),
            'rating': self._get_column_value(row, ['rating', 'score', 'stars']),
        }
        
        # Remove empty metadata
        metadata = {k: v for k, v in metadata.items() if v}
        
        return {
            'content': content,
            'metadata': metadata
        }
    
    # Extract content from the row using a priority order of possible column names, with fallbacks to combine text fields if no dedicated content column is found.
    def _extract_content(self, row: pd.Series) -> str:
        # Priority order for content columns
        content_cols = [
            'description', 'content', 'text', 'about', 'details', 'overview',
            'summary', 'info', 'information', 'full_description', 'main_text'
        ]
        
        for col in content_cols:
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        
        # If no dedicated content column, combine long text fields
        text_parts = []
        for col in row.index:
            val = row[col]
            if pd.notna(val) and isinstance(val, str):
                # Only include substantial text fields
                if len(val) > 100 and not val.startswith('http'):
                    text_parts.append(val)
        
        return ' '.join(text_parts) if text_parts else ''
    
    # Get a single column value with multiple possible matching names
    def _get_column_value(self, row: pd.Series, possible_names: List[str]) -> Optional[str]:
        for name in possible_names:
            if name in row.index and pd.notna(row[name]):
                val = str(row[name]).strip()
                if val and val.lower() not in ['nan', 'none', 'null', '']:
                    return val
        return None
    
    # Parse a list field that may contain multiple items 
    def _parse_list_field(self, row: pd.Series, possible_names: List[str]) -> List[str]:
        raw_value = self._get_column_value(row, possible_names)
        if not raw_value:
            return []
        
        # Split by common delimiters
        items = re.split(r'[,;|]', raw_value)
        return [item.strip() for item in items if item.strip()]
    
    # Clean and normalize text content.
    def _clean_text(self, text: str) -> str:
        if not text:
            return ''
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?\-\'\"()]', ' ', text)
        
        return text.strip()
    
    # Get statistics about the loaded documents, including counts, average length, category distribution, and unique locations.
    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}
        
        # Basic statistics
        stats = {
            'total_documents': len(self.documents),
            'total_chars': sum(len(d['content']) for d in self.documents),
            'avg_doc_length': sum(len(d['content']) for d in self.documents) / len(self.documents),
            'categories': {},
            'locations': set(),
            'source_files': set()
        }
        
        # Analyze metadata for category distribution, unique locations, and source files.
        for doc in self.documents:
            meta = doc['metadata']
            cat = meta.get('category', 'unknown')
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
            if meta.get('location'):
                stats['locations'].add(meta['location'])
            stats['source_files'].add(meta.get('source_file', 'unknown'))
        
        # Convert sets to counts and lists for output
        stats['unique_locations'] = len(stats['locations'])
        stats['locations'] = sorted(list(stats['locations']))[:20]  # Limit output
        stats['source_files'] = list(stats['source_files'])
        
        return stats
    
    # Save processed documents to a JSON file.
    def save_processed(self, output_dir: str = "data/processed"):
        import json
        from datetime import datetime
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save with a timestamp to avoid overwriting previous files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"processed_documents_{timestamp}.json"
        
        # Save documents to JSON file with pretty formatting
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Log the save operation
        logger.info(f"Saved processed documents to {filename}")
        return filename