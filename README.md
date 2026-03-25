# LLLMS Local Large Language Models

## Before You Begin

You should know that nearly all this python code was written by [Ambika Joshi (aka Computational Mama)](https://computationalmama.xyz/) who will also be joining our class as a guest lecturer on April 20th :-)

### What is Ollama?
Ollama lets you run AI models locally on your own computer, meaning your data never leaves your machine and you don't need an internet connection or API key to use it.

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running

---

## Step 1: Open Your Terminal

The terminal is a text-based way to control your computer by typing commands. 

### On Mac
1. Press **Command (⌘) + Space** to open Spotlight Search
2. Type **Terminal** and press Enter
3. A window with a text prompt will open — this is your terminal

### On Windows
1. Press the **Windows key**, type **cmd**, and press Enter
2. A black window with a text prompt will open — this is your Command Prompt

---

## Step 2: Navigate to Your Project Folder

Once your terminal is open, you need to navigate to the folder where you downloaded this project. This is done with the `cd` command (short for "change directory").

Think of it like clicking into folders on your Desktop, but with text.

**Example — if your project is in your Downloads folder:**

On Mac:
```bash
cd Downloads/local_rag/python
```

On Windows:
```bash
cd Downloads\local_rag\python
```

> 💡 **Tip:** You can drag a folder from Finder (Mac) or File Explorer (Windows) directly into your terminal window and it will automatically type the path for you!

> 💡 **Tip:** Press **Tab** while typing a folder name to autocomplete it.

---

## Step 3: Make Sure Python is Installed

### On Mac
In your terminal, run:
```bash
python3 --version
```
If you see `Python 3.x.x` you're good. If not, download it from **https://python.org/downloads** and install it. On Mac you will always type `python3` and `pip3` instead of `python` and `pip`.

### On Windows
In your Command Prompt, run:
```bash
python --version
```
If you see `Python 3.x.x` you're good. If not:
1. Download Python from **https://python.org/downloads**
2. Run the installer — **make sure to check "Add Python to PATH"** before clicking Install (this is the most common mistake!)
3. Close and reopen Command Prompt, then run `python --version` again to confirm

---

## Step 4: Install Ollama

Go to **https://ollama.com/download** and download the installer for your operating system. Run it and follow the instructions.

Once installed, start the Ollama server. Keep this running in a terminal window in the background while you use the app:
```bash
ollama serve
```

> 💡 You'll need to keep this terminal window open the whole time. Open a **new** terminal window for the next steps.

> 💡 On Mac you can also run `brew services start ollama` to have it start automatically every time you log in, so you never have to think about it.

---

## Step 5: Pull Ollama Models

In a new terminal window, download the AI models the app needs:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

This may take a few minutes depending on your internet speed.

---

## Step 6: Install Python Dependencies

Navigate to your project folder (see Step 2), then run:

**On Mac & PC:**
```bash
pip3 install -r requirements.txt
```


---

## Step 7: Build the Database

Drop your PDFs into the `./docs/` folder first, then:

**On Mac & PC:**
```bash
python3 rag.py build
```


---

## Step 8: Chat!

**On Mac & PC:**
```bash
python rag.py
```

---

## Troubleshooting

**"No database found"**
You need to build the database first. Run:
```bash
python3 rag.py build   # Mac
python rag.py build    # Windows
```

**"Connection refused" / Ollama not responding**
Ollama is not running. Open a new terminal window and run:
```bash
ollama serve
```

**"Address already in use"**
```bash
lsof -i :6600                    # Mac
netstat -ano | findstr :6600     # Windows
```

**Slow responses**
- Switch to a smaller model (see Customization section below)
- Reduce `n_results` in the query (see Customization section below)

---

## Customization

### Change the model

You'll need a code editor (e.g. VSCode) to edit `rag.py`. Inside `SimpleRAG.__init__()`:
```python
self.llm = Ollama(model="llama3.2:1b")   # faster, less accurate
self.llm = Ollama(model="llama3.1:8b")   # slower, more accurate
```

---

### Change chunk size

**What is a chunk?**
When your PDFs are loaded, they're too long to process all at once. The RAG system splits them into smaller pieces called **chunks** — think of it like cutting a textbook into index cards. Each chunk is a snippet of text that gets stored and searched independently.

`chunk_size` controls how many characters are in each piece. `chunk_overlap` makes consecutive chunks share a little text at the edges, so nothing gets cut off mid-sentence.
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # number of characters per chunk (try 200–1000)
    chunk_overlap=50   # characters shared between neighboring chunks
)
```

- **Smaller chunks** → more precise but may lose context
> 💡 **Why?** Imagine cutting a paragraph into single sentences. Each sentence on its own might not make sense without the ones around it. If the answer to a question spans multiple sentences, the AI may only retrieve one piece and miss the full meaning. `chunk_overlap` helps with this by repeating a little text between chunks.

- **Larger chunks** → more context but slower and may confuse the model
> 💡 **Why?** Every AI model has a limit on how much text it can process at once, called a **context window**. If your chunks are too large, they eat up that space quickly, leaving the model less room to reason and respond. Smaller chunks are also easier for the model to match precisely to your question.

---

### Change number of retrieved chunks

When you ask a question, the system finds the most relevant chunks from your documents and passes them to the AI. `n_results` controls how many chunks it retrieves.
```python
results = collection.query(
    query_embeddings=[q_embedding],
    n_results=3   # how many chunks to send to the AI (try 3–10)
)
```

- **Fewer chunks** → faster, more focused answer
- **More chunks** → more context, better for complex questions, but slower

> 💡 **Why not always use fewer chunks?** Think of it like giving someone research notes before asking them a question. If you hand them 3 relevant index cards they'll answer quickly and clearly. If you hand them 20 cards — some relevant, some not — it takes longer and they might get sidetracked by irrelevant information. But if the answer to your question is spread across multiple documents or sections, retrieving only 1 or 2 chunks might mean the AI never even sees the relevant information. For example, asking *"summarize the key themes across all the readings"* needs more chunks than asking *"what is the definition of X"*.