import logging
from abc import ABC, abstractmethod
from overrides import override
from config.conf import Config

log = logging.getLogger("blip.retriever")


class Retriever(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def index(
            self,
            course_name: str,
            collection_name: str,
            chunk_ids: list[str],
            texts: list[str],
            metadatas: list[dict],
    ) -> None:
        """Upsert chunks into the specified course and collection."""
        ...

    @abstractmethod
    def delete(
            self, course_name: str, collection_name: str, chunk_ids: list[str]
    ) -> None:
        """Delete specific chunks. Must raise on failure: callers record the deletion only if it happened."""
        ...

    @abstractmethod
    def search(
            self, course_name: str, collection_name: str, query: str, top_k: int = 3
    ) -> list[dict]:
        """
        Retrieve the top-k chunks. Each hit: text, source, heading (as before), plus id, distance,
        content_item_id and version -- enough to record exactly which file version was used.
        """
        ...


class ChromaRetriever(Retriever):
    def __init__(self, config: Config, embedding_fn=None):
        """
        One persistent Chroma store per course (config.chromadb_dir(course)), several collections
        per store. embedding_fn defaults to the sentence-transformer named in the config.
        """
        super().__init__(config)
        if embedding_fn is None:
            from chromadb.utils import embedding_functions
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.blip_embed_model)
        self.embedding_fn = embedding_fn

        # Connection pools to reuse clients and collections across multiple courses
        self._clients = {}
        self._collections = {}

    def _get_collection(self, course_name: str, collection_name: str):
        """Helper to resolve and cache the exact client and collection on demand."""
        import chromadb
        from chromadb.config import Settings

        # 1. Resolve Client (one isolated store per course). chromadb_dir() fills the
        #    {course_tag} template; joining the raw template created a literal "{course_tag}" dir.
        if course_name not in self._clients:
            db_path = self.config.chromadb_dir(course_name)
            db_path.mkdir(parents=True, exist_ok=True)
            self._clients[course_name] = chromadb.PersistentClient(
                path=str(db_path), settings=Settings(anonymized_telemetry=False)
            )
            log.info(f"Initialized ChromaDB client for course: {course_name}")

        # 2. Resolve Collection (Multiple collections per course)
        cache_key = f"{course_name}_{collection_name}"
        if cache_key not in self._collections:
            self._collections[cache_key] = self._clients[
                course_name
            ].get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(
                f"Initialized collection '{collection_name}' for course '{course_name}'"
            )

        return self._collections[cache_key]

    @override
    def index(
            self,
            course_name: str,
            collection_name: str,
            chunk_ids: list[str],
            texts: list[str],
            metadatas: list[dict],
    ) -> None:
        if not chunk_ids:
            return

        collection = self._get_collection(course_name, collection_name)
        try:
            collection.upsert(ids=chunk_ids, documents=texts, metadatas=metadatas)
            log.debug(
                f"Successfully indexed {len(chunk_ids)} chunks into {course_name}/{collection_name}."
            )
        except Exception as e:
            log.error(
                f"Failed to index chunks into {course_name}/{collection_name}: {e}"
            )
            raise

    @override
    def delete(
            self, course_name: str, collection_name: str, chunk_ids: list[str]
    ) -> None:
        if not chunk_ids:
            return

        collection = self._get_collection(course_name, collection_name)
        try:
            collection.delete(ids=chunk_ids)
            log.debug(
                f"Successfully deleted {len(chunk_ids)} stale chunks from {course_name}/{collection_name}."
            )
        except Exception as e:
            # Re-raised: swallowing this left outdated slides retrievable while the index
            # recorded them as deleted.
            log.error(
                f"Failed to delete chunks from {course_name}/{collection_name}: {e}"
            )
            raise

    @override
    def search(
            self, course_name: str, collection_name: str, query: str, top_k: int = 3
    ) -> list[dict]:
        collection = self._get_collection(course_name, collection_name)

        results = collection.query(query_texts=[query], n_results=top_k,
                                   include=["documents", "metadatas", "distances"])

        formatted_results = []
        if results.get("documents") and results["documents"][0]:
            for chunk_id, doc, meta, dist in zip(results["ids"][0], results["documents"][0],
                                                 results["metadatas"][0], results["distances"][0]):
                meta = meta or {}
                formatted_results.append(
                    {
                        "text": doc,
                        "source": meta.get("filename", "Unknown"),
                        "heading": meta.get("heading", ""),
                        "id": chunk_id,
                        "distance": dist,
                        "content_item_id": meta.get("content_item_id"),
                        "version": meta.get("version"),
                    }
                )

        return formatted_results
