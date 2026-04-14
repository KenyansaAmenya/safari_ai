import tiktoken

from src.config import settings
from src.domain.models import Chunk, Document

class SlidingWindowChunker:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = settings.chunk_size
        self.overlap = settings.chunk_overlap

    def chunk(self, document: document) -> list[chunk]:
        text = document.content
        tokens = self.encoder.encode(text)

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            # find character position
            char_start = len(self.encoder.decode(tokens[:start]))
            char_end = char_start + len(chunk_text)

            chunk = chunk(
                document_id=document.id or document.source_name,
                text=chunk_text,
                chunk_index=chunk_idx,
                start_pos=char_start,
                end_pos=char_end,
                metadata={
                    "source": document.source_name,
                    "location": document.location,
                    "category": document.category,
                    "chunk_index": chunk_idx
                }
            )    
            chunks.append(chunk)

            # slide window
            start += (self.chunk_size - self.overlap)
            chunk_idx += 1

            # avoid infinite loop on last chunk
            if end == len(tokens):
                break

        return chunks    