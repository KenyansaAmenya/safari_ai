from enum import Enum

class category(str, Enum):
    WILDLIFE = "wildlife"
    BEACH = "beach"
    CULTURE = "culture"
    ADVENTURE = "adventure"
    LUXURY = "luxury"
    BUDGET = "budget"
    FOOD = "food"
    HIKING =  "hiking"
    BIRDING = "birding"
    GENERAL = "general"

    class location(str, enum):
        MAASAI_MARA = "maasai_mara"
        MOMBASA = "mombasa"
        MOUNT_KENYA = "mount_kenya"
        LAKE_NAKURU = "lake_nakuru"
        AMBOSELI = "amboseli"
        TSAVO = "tsavo"
        LAMU = "lamu"
        NAIROBI = "nairobi"
        DIANI = "diani"
        WATAMU = "watamu"
        GENERAL = "general"