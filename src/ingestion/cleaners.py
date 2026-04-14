import re
import html

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        text = html.unescape(text)                    # Step 1: Decode HTML entities
        text = re.sub(r'<[^>]+>', ' ', text)          # Step 2: Remove HTML tags
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)     # Step 3: Remove special chars
        text = re.sub(r'\s+', ' ', text)              # Step 4: 
        text = re.sub(r'\n+', '\n', text)
        return text.strip().lower()

    @staticmethod
    def normalize_price(text: str) -> str:
        text = re.sub(r'kshs?\.?\s*(\d+)', r'kes \1', text, flags=re.I)
        text = re.sub(r'usd\s*\$?\s*(\d+)', r'usd \1', text, flags=re.I)
        return text

    @staticmethod
    def normalize_location(text: str) -> str:
        location_map = {
            'masai mara': 'maasai_mara',
            'mara': 'maasai_mara',
            'mt kenya': 'mount_kenya',
            'lake nakuru national park': 'lake_nakuru',
        }       
        text_lower = text.lower()
        for old, new in location_map.items():
            text_lower = text_lower.replace(old, new)
        return text_lower    