from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")
                conn.commit()
                print("✅ Table truncated successfully")
        finally:
            conn.close()

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        """Save a text chunk with its embedding to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Convert embedding list to string format for PostgreSQL vector
                embedding_str = str(embedding)

                query = """
                    INSERT INTO vectors (document_name, text, embedding)
                    VALUES (%s, %s, %s::vector)
                """
                cursor.execute(query, (document_name, text, embedding_str))
                conn.commit()
        finally:
            conn.close()

    def process_text_file(
        self,
        file_path: str,
        chunk_size: int = 300,
        overlap: int = 40,
        dimensions: int = 1536,
        truncate: bool = False
    ):
        """
        Process a text file: load, chunk, embed, and store in database

        Args:
            file_path: Path to the text file to process
            chunk_size: Size of text chunks (default: 300)
            overlap: Overlap between chunks (default: 40)
            dimensions: Embedding dimensions (default: 1536)
            truncate: Whether to truncate the table before inserting (default: False)
        """
        # Truncate table if requested
        if truncate:
            self._truncate_table()

        # Load the file content
        print(f"📖 Loading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Generate chunks
        print(f"✂️  Chunking text (chunk_size={chunk_size}, overlap={overlap})")
        chunks = chunk_text(content, chunk_size, overlap)
        print(f"✅ Generated {len(chunks)} chunks")

        # Generate embeddings for all chunks
        print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        embeddings_dict = self.embeddings_client.get_embeddings(chunks, dimensions=dimensions)
        print(f"✅ Embeddings generated")

        # Save chunks with embeddings to database
        print(f"💾 Saving chunks to database...")
        document_name = file_path.split('/')[-1].split('\\')[-1]  # Extract filename

        for idx, chunk in enumerate(chunks):
            embedding = embeddings_dict[idx]
            self._save_chunk(document_name, chunk, embedding)

        print(f"✅ Successfully processed and stored {len(chunks)} chunks from {document_name}")

    def search(
        self,
        query: str,
        search_mode: SearchMode = SearchMode.COSINE_DISTANCE,
        top_k: int = 5,
        min_score: float = 0.5,
        dimensions: int = 1536
    ) -> list[str]:
        """
        Search for relevant text chunks using semantic similarity

        Args:
            query: The search query text
            search_mode: Distance metric to use (cosine or euclidean)
            top_k: Number of top results to return (default: 5)
            min_score: Minimum similarity threshold (default: 0.5)
            dimensions: Embedding dimensions (default: 1536)

        Returns:
            List of relevant text chunks
        """
        # Generate embedding for the query
        embeddings_dict = self.embeddings_client.get_embeddings([query], dimensions=dimensions)
        query_embedding = embeddings_dict[0]
        query_embedding_str = str(query_embedding)

        # Choose the distance operator based on search mode
        distance_operator = "<=>" if search_mode == SearchMode.COSINE_DISTANCE else "<->"

        # Build and execute the search query
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query_sql = f"""
                    SELECT text, (embedding {distance_operator} %s::vector) as distance
                    FROM vectors
                    WHERE (embedding {distance_operator} %s::vector) < %s
                    ORDER BY distance
                    LIMIT %s
                """

                # Calculate threshold based on min_score
                # For cosine distance: 0 = identical, 2 = opposite
                # For euclidean: smaller = more similar
                threshold = 1 - min_score if search_mode == SearchMode.COSINE_DISTANCE else min_score

                cursor.execute(query_sql, (query_embedding_str, query_embedding_str, threshold, top_k))
                results = cursor.fetchall()

                # Extract text from results
                return [row['text'] for row in results]
        finally:
            conn.close()
