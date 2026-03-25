#!/usr/bin/env python3
"""
Simple Local RAG - 100% Offline
No server, no internet required after setup
"""

import chromadb
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os

class SimpleRAG:
    def __init__(self):
        self.db_path = "./rag_database"
        self.docs_path = "./docs"

        # Local Ollama
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = Ollama(model="qwen2.5:7b")

        # Local ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)

    def build(self):
        """Build database from PDFs (run once)"""
        print("📂 Loading PDFs...")
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs = loader.load()

        if not docs:
            print(f"❌ No PDFs found in {self.docs_path}/")
            return

        print(f"✂️  Splitting {len(docs)} documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        print(f"🧮 Creating embeddings for {len(chunks)} chunks...")
        collection = self.client.get_or_create_collection("docs")

        # Clear existing data
        try:
            collection.delete(ids=[f"chunk_{i}" for i in range(10000)])
        except:
            pass

        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_documents([chunk.page_content])[0]
            collection.add(
                ids=[f"chunk_{i}"],
                documents=[chunk.page_content],
                embeddings=[embedding],
                metadatas=[chunk.metadata or {}]
            )

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(chunks)}")

        print(f"✅ Database built! {len(chunks)} chunks ready")

    def ask(self, question, show_sources=False):
        """Ask a question"""
        collection = self.client.get_collection("docs")

        # Embed question
        q_embedding = self.embeddings.embed_documents([question])[0]

        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=3
        )

        if not results['documents'][0]:
            return "No relevant information found."

        # Build context
        context = "\n\n".join(results['documents'][0])

        # Generate answer
        prompt = f"""Use this context to answer. If unsure, say so.

Context:
{context}

Question: {question}

Answer:"""

        answer = self.llm.invoke(prompt)

        if show_sources:
            sources = [m.get('source', 'Unknown') for m in results['metadatas'][0]]
            return f"{answer}\n\n📚 Sources: {', '.join(set(sources))}"

        return answer

    def stats(self):
        """Show database stats"""
        try:
            collection = self.client.get_collection("docs")
            count = collection.count()
            print(f"📊 Database has {count} chunks")
        except:
            print("❌ No database found. Run build() first.")

# ============ MAIN ============
if __name__ == "__main__":
    import sys

    rag = SimpleRAG()

    # Command line interface
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "build":
            rag.build()

        elif command == "ask":
            if len(sys.argv) < 3:
                print("Usage: python rag.py ask 'your question'")
            else:
                question = " ".join(sys.argv[2:])
                print(f"\n❓ {question}\n")
                answer = rag.ask(question, show_sources=True)
                print(f"💡 {answer}\n")

        elif command == "stats":
            rag.stats()

        else:
            print("Commands: build, ask, stats")

    else:
        # Interactive mode
        print("\n🤖 Simple Local RAG")
        print("=" * 50)

        rag.stats()

        print("\n💬 Ask questions (or 'quit' to exit)\n")

        while True:
            try:
                question = input("❓ ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Bye!")
                    break

                if question.lower() == 'stats':
                    rag.stats()
                    continue

                answer = rag.ask(question, show_sources=True)
                print(f"\n💡 {answer}\n")
                print("-" * 50 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Bye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
