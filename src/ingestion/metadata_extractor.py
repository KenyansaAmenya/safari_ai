# pattern based detection
import re
from typing import Optional
from src.domain.enums import Category, Location

class MetadataExtractor:
    LOCATION_PATTERNS = {
        Location.MAASAI_MARA: ['maasai mara', 'masai mara', 'mara', 'big five'],
        Location.MOMBASA: ['mombasa', 'coastal city', 'fort jesus', 'old town'],
        Location.MOUNT_KENYA: ['mount kenya', 'mt kenya', 'batian', 'nelion'],
        Location.LAKE_NAKURU: ['lake nakuru', 'nakuru', 'flamingo', 'rhino sanctuary'],
        Location.AMBOSELI: ['amboseli', 'kilimanjaro view', 'elephant'],
        Location.TSAVO: ['tsavo', 'red elephant', 'man-eaters'],
        Location.LAMU: ['lamu', 'swahili culture', 'donkey', 'old town'],
        Location.NAIROBI: ['nairobi', 'capital', 'giraffe centre', 'nairobi national park'],
        Location.DIANI: ['diani', 'diani beach', 'south coast'],
        Location.WATAMU: ['watamu', 'marine park', 'turtle'],
    }
    
    CATEGORY_PATTERNS = {
        Category.WILDLIFE: ['wildlife', 'safari', 'big five', 'lion', 'elephant', 'leopard', 'buffalo', 'rhino'],
        Category.BEACH: ['beach', 'ocean', 'swim', 'snorkel', 'dive', 'coast', 'sand'],
        Category.CULTURE: ['culture', 'maasai', 'swahili', 'village', 'traditional', 'heritage'],
        Category.ADVENTURE: ['adventure', 'hike', 'trek', 'climb', 'rafting', 'bungee'],
        Category.LUXURY: ['luxury', '5-star', 'resort', 'spa', 'exclusive', 'high-end'],
        Category.BUDGET: ['budget', 'cheap', 'affordable', 'backpacker', 'hostel'],
        Category.FOOD: ['food', 'cuisine', 'restaurant', 'dining', 'swahili food', 'nyama choma'],
        Category.HIKING: ['hike', 'trek', 'trail', 'summit', 'climb', 'mountain'],
        Category.BIRDING: ['bird', 'birding', 'ornithology', 'flamingo', 'eagle', 'stork'],
    }

    def extract_location(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        scores = {}

        for location, patterns in self.LOCATION_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            if score > 0:
                scores[location] = score

        if scores:
            return max(scores, key=scores.get).value
        return none

    def extract_category(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        scores = {}

        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            if score > 0:
                scores[category] = score

            if scores:
                return max(scores, key=scores.get).value
            return none

    def extract_price_range(self, text: str) -> Optional[str]:
        text_lower = text.lower()

        luxury = ['luxury', '5-star', 'expensive', 'high-end', 'premium', '$300', 'kes 50000']
        budget = ['budget', 'cheap', 'affordable', 'low-cost', 'backpacker', '$50', 'kes 5000']   

        lux_score = sum(1 for w in luxury if w in text_lower)
        bud_score = sum(1 for w in budget if w in text_lower)  

        if lux_score> bud_score:
            return "luxury"
        elif bud_score > lux_score:
            return "budget"
        return "moderate" if "price" in text_lower or "cost" in text_lower else None                  

    
    