#!/usr/bin/env python3
"""
Simple Local RAG - 100% Offline
No server, no internet required after setup
"""

import chromadb
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    CSVLoader,
)
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer

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
        loaders = [
            DirectoryLoader(self.docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(
                self.docs_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            ),
            DirectoryLoader(
                self.docs_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            ),
            DirectoryLoader(self.docs_path, glob="**/*.csv", loader_cls=CSVLoader),
        ]
        docs = [doc for loader in loaders for doc in loader.load()]

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

    def serve(self, port=8000):
        """Serve web/index.html and answer questions at /ask (stdlib only)"""
        rag = self
        web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                name = self.path.split("?")[0].lstrip("/") or "index.html"
                path = os.path.normpath(os.path.join(web_dir, name))
                if not path.startswith(web_dir + os.sep) or not os.path.isfile(path):
                    return self._send(404, b"Not found", "text/plain")
                ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                with open(path, "rb") as f:
                    self._send(200, f.read(), ctype)

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    question = json.loads(self.rfile.read(length)).get("question", "").strip()
                    result = str(rag.ask(question, show_sources=True))
                    answer, _, sources = result.partition("\n\n📚 Sources:")
                    out = {"answer": answer.strip(), "sources": sources.strip()}
                except Exception as e:
                    out = {"error": str(e)}
                self._send(200, json.dumps(out).encode(), "application/json")

            def log_message(self, *args):
                pass  # keep the terminal quiet

        print(f"🌐 Chat running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()

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

        elif command == "serve":
            rag.serve()

        else:
            print("Commands: build, ask, stats, serve")

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