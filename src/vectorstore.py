import os
import shutil
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
import time


DB_PATH = "./data/chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"


class VectorStoreManager:
    def __init__(self, collection_name: str = "candidate_data"):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.collection_name = collection_name
        self.collection_path = os.path.join(DB_PATH, collection_name)
        self.vector_store: Optional[Chroma] = None

    def reset_db(self):
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                print(f"[INFO] Collezione {self.collection_name} resettata.")
            except Exception as e:
                print(f"[WARN] Errore nel reset collezione: {e}")
            self.vector_store = None

            if os.path.exists(self.collection_path):
                shutil.rmtree(self.collection_path)
                print(
                    f"[INFO] Cartella e collezione '{self.collection_name}' eliminate fisicamente da {self.collection_path}.")

    def create_store(self, documents: List[Any]):
        if not documents:
            return None


        self.reset_db()

        os.makedirs(self.collection_path, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.collection_path,
            collection_name=self.collection_name
        )
        return self.vector_store

    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        if not self.vector_store:
            if os.path.exists(self.collection_path):
                self.vector_store = Chroma(
                    persist_directory=self.collection_path,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            else:
                raise ValueError("Vector store non inizializzato. Chiama create_store prima.")

        kwargs = search_kwargs or {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def get_vectorstore(self):
        if not self.vector_store:
            if os.path.exists(self.collection_path):
                self.vector_store = Chroma(
                    persist_directory=self.collection_path,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            else:
                raise ValueError("Vector store non inizializzato. Chiama create_store prima.")

        return self.vector_store


    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.vector_store:
            try:
                self.get_retriever()
            except ValueError:
                return []

        docs = self.vector_store.similarity_search(query, k=k, filter=filter)

        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    def create_chunks(self, text: str, metadata: dict) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            add_start_index=True
        )
        return [Document(page_content=chunk, metadata=metadata) for chunk in text_splitter.split_text(text)]

    def is_collection_valid(self, current_hash: str) -> bool:
        hash_path = os.path.join(self.collection_path, "collection_hash.json")

        if not os.path.exists(hash_path) or not os.path.exists(self.collection_path):
            return False

        try:
            with open(hash_path, "r") as f:
                cached_data = json.load(f)
                return cached_data.get("hash") == current_hash
        except:
            return False

    def save_collection_metadata(self, current_hash: str):
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path, exist_ok=True)

        hash_path = os.path.join(self.collection_path, "collection_hash.json")
        with open(hash_path, "w") as f:
            json.dump({"hash": current_hash, "timestamp": time.time()}, f)

    def add_documents(self, documents: List[Document]):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.collection_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        self.vector_store.add_documents(documents)

    def get_all_documents(self) -> list[Document]:
        try:
            results = self.vector_store._collection.get()

            all_docs = []
            if results and "documents" in results:
                for i in range(len(results["documents"])):
                    testo = results["documents"][i]
                    metadata = results["metadatas"][i] if results.get("metadatas") else {}

                    all_docs.append(Document(page_content=testo, metadata=metadata))

            return all_docs

        except Exception as e:
            print(f"[ERROR] Impossibile estrarre i documenti da Chroma: {e}")
            return []


vstore_manager = VectorStoreManager()
vstore_cache = VectorStoreManager(collection_name="extracted_skills_cache")
vstore_rag = VectorStoreManager(collection_name="rag_corpus")