import vecs
from src.config import settings

class VecsClient:
    def __init__(self):
        self.client = vecs.create_client(settings.supabase_db_connection)
        self.collection_name = settings.vector_collection
        self.dimension = settings.embedding_dimension

    def get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            dimension=self.dimension
        )    

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)    