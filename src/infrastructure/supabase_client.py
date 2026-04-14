from supabase import create_client, client
from src.config import settings

class SupabaseClient:
    _instance: Client = None

    @classmethod
    def get_client(cls) -> Client:
        if cls._instance is None: 
            cls._instance = create_client(
                settings.supabase_url,
                settings.supabase_service_key
            )
        return cls._instance

def get_supabase() -> Client:
    return SupabaseClient.get_client()            