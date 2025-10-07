import os
from pathlib import Path
from PIL import Image
import os

import llama_cpp
def safe_del(self):
    try:
        if hasattr(self, "_ctx") and self._ctx is not None:
            self.close()
    except Exception:
        pass
llama_cpp.Llama.__del__ = safe_del

# --- LangChain Imports ---
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- VLM (BLIP) Imports ---
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- PDF Parsing ---
import fitz  # PyMuPDF

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
PDF_FILE = DATA_DIR / "research_paper.pdf"  # Place your PDF here
LLM_MODEL_PATH = "models/phi-2.Q5_K_S.gguf"  # Path to GGUF model -> https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q5_K_S.gguf

# Embedding & Vision-Language Models (cached)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ============================================================
# BLIP IMAGE CAPTIONING CLASS
# ============================================================

class BlipCaptioner:
    """A wrapper for the BLIP Vision-Language Model to generate image captions."""
    def __init__(self, model_name: str, device: str = "cpu"):
        print(f"-> Loading BLIP model ({model_name}) on {device}...")

        # Use local cache
        self.processor = BlipProcessor.from_pretrained(model_name, cache_dir=str("blip"), use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, cache_dir=str("blip")).to(device)
        self.device = device

    def generate_caption(self, image_path: str) -> str:
        """Generate a caption for a single image file."""
        try:
            raw_image = Image.open(image_path).convert("RGB")
            prompt = "A detailed caption for a figure from a research paper: "
            inputs = self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=150)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error captioning image: {e}"

# ============================================================
# PDF INGESTION & IMAGE CAPTIONING PIPELINE
# ============================================================

def ingestion_pipeline(pdf_path: Path, captioner: BlipCaptioner):
    """Extracts text and images from PDF using PyMuPDF, generates captions, and builds documents."""
    temp_dir = DATA_DIR / "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    print("1. Extracting text and images from PDF using PyMuPDF...")
    all_docs = []
    text_content = []

    # --- Open PDF ---
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text("text")
        if text.strip():
            text_content.append(f"Page {page_num + 1}: {text.strip()}")

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image_path = temp_dir / image_filename
            with open(image_path, "wb") as f:
                f.write(image_bytes)

    doc.close()

    # --- Split Text into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for i, chunk in enumerate(text_splitter.split_text(" ".join(text_content))):
        all_docs.append(Document(
            page_content=chunk,
            metadata={"source": str(pdf_path), "type": "text", "chunk_id": i}
        ))

    # --- Caption Images ---
    image_paths = [temp_dir / f for f in os.listdir(temp_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"2. Generating captions for {len(image_paths)} images...")
    
    saved_img_dir = Path("saved_figures")
    saved_img_dir.mkdir(exist_ok=True)

    for i, path in enumerate(image_paths):
        caption = captioner.generate_caption(str(path))
        # Copy image to "saved_figures" directory instead of deleting
        new_path = saved_img_dir / f"figure_{i+1}_{os.path.basename(path)}"
        os.replace(path, new_path)

        caption_doc = Document(
            page_content=f"VISUAL CONTEXT (Figure {i + 1}): {caption}",
            metadata={
                "source": str(pdf_path),
                "type": "image_caption",
                "image_file": str(new_path)
            }
        )
        all_docs.append(caption_doc)

    os.rmdir(temp_dir)

    print(f"-> Total documents created (text + captions): {len(all_docs)}")
    return all_docs

# ============================================================
# RAG PIPELINE SETUP
# ============================================================

from pathlib import Path

def setup_rag_chain(documents: list[Document]):
    """Creates a LangChain RAG pipeline using LlamaCpp and FAISS."""
    print("\n3. Initializing LlamaCpp and FAISS...")

    # --- Load Llama Model ---
    llm = LlamaCpp(
        model_path=str(LLM_MODEL_PATH),
        temperature=0.35,       # ✅ balanced: factual but not dry
        top_p=0.92,             # keeps diversity slightly higher without chaos
        repeat_penalty=1.15,    # prevents repetitive phrasing
        max_tokens=3072,        # large enough for structured summaries
        n_ctx=2048,             # if your model supports it (increases context window)
        n_batch=256,            # fine balance between speed and stability
        n_gpu_layers=0,         # ✅ use GPU acceleration if available
        n_threads=max(1, os.cpu_count() - 1),
        verbose=False,
        model_kwargs={
            "frequency_penalty": 0.4,  # discourage repeating same ideas
            "presence_penalty": 0.3,   # encourage new points slightly
        }
    )

    # --- Load Embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        cache_folder="embeddings"
    )

    # --- FAISS Index Directory ---
    faiss_dir = Path("faiss_index")
    faiss_dir.mkdir(exist_ok=True)

    if (faiss_dir / "index.faiss").exists() and (faiss_dir / "index.pkl").exists():
        print(f"-> Found existing FAISS index at {faiss_dir}, loading it...")
        vectorstore = FAISS.load_local(
            str(faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("4. Building new FAISS vector store from documents...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(faiss_dir))
        print(f"✅ FAISS index saved successfully at: {faiss_dir}")

    # --- Build Retriever ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # smaller context for speed

    # --- Define Custom Prompt ---
    prompt = PromptTemplate(
        template=(
            "You are an expert scientific summarizer. Based on the CONTEXT below (text and figure captions), "
            "produce a detailed structured summary of the research paper using the following exact Markdown format:\n\n"
            "## Summary\n"
            "- Write a concise paragraph (150–300 words) summarizing the main topic, methods, and findings.\n\n"
            "## Highlights\n"
            "- Write **exactly 5 to 7** concise bullet points summarizing the most important results, techniques, or findings.\n"
            "- Start each highlight with a clear, strong statement.\n\n"
            "## Key Insights\n"
            "- Write **exactly 5 to 7** detailed key insights.\n"
            "- Each insight should begin with a **bolded title** (e.g., '**Impact of Light Intensity:** ...') followed by a short analytical explanation.\n\n"
            "Do NOT include code, explanations, or any text outside this Markdown structure.\n"
            "Continue until all 5–7 highlights and 5–7 key insights are written.\n"
            "If there are fewer than 5, infer additional reasonable insights from context.\n"
            "Always output in Markdown.\n\n"
            "-------------------\n"
            "CONTEXT:\n"
            "{context}\n"
            "-------------------\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    # --- Build the RAG Chain ---
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    print("✅ RAG chain initialized successfully.")
    return rag_chain

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import time
    start_time = time.time()
    if not PDF_FILE.exists():
        print(f"❌ ERROR: PDF file not found at {PDF_FILE}. Please place your 'research_paper.pdf' in the 'data' folder.")
    elif not os.path.exists(LLM_MODEL_PATH):
        print(f"❌ ERROR: LLM model not found at {LLM_MODEL_PATH}. Please download your GGUF file and rename it accordingly.")
    else:
        # Step 1: Initialize captioner
        blip_captioner = BlipCaptioner(BLIP_MODEL_NAME)

        # Step 2: Parse PDF and caption images
        documents = ingestion_pipeline(PDF_FILE, blip_captioner)

        # Step 3: Set up RAG pipeline
        rag_chain = setup_rag_chain(documents)

        # Step 4: Run query
        print("\n5. Running Multimodal RAG Query...")
        query = (
            "Summarize the provided document and generate a structured Markdown report with:\n"
            "1. A single concise summary paragraph.\n"
            "2. 5–7 bullet points under '## Highlights'.\n"
            "3. 5–7 detailed analytical points under '## Key Insights'.\n"
            "Follow Markdown format exactly, no code or extra commentary."
        )
        response = rag_chain.invoke({"query": query})

        print("\n\n" + "=" * 20 + " FINAL GENERATED OUTPUT " + "=" * 20)
        print(response["result"])
        print("=" * 64)
        try:
            del rag_chain.llm
            import gc
            gc.collect()
            print("🧹 LlamaCpp model cleaned up safely.")
        except Exception:
            pass
        end_time = time.time()
        print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")

