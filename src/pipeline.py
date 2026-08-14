"""
Document Q&A Pipeline — Production RAG Implementation.

This module retrieves relevant marketing agency context using FAISS vector search
and generates accurate answers via a local Hugging Face Flan-T5 model.

Useful docs:
  - Vector store search: https://python.langchain.com/docs/how_to/vectorstores/
  - Hugging Face pipelines: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
"""

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List

# Suppress noisy HuggingFace/transformers warnings for clean CLI output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.knowledge_base import build_knowledge_base

# Optional rich UI formatting for enhanced terminal experience
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


# ──────────────────────────────────────────────
# Provided: local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(prompt: str) -> List[Dict[str, str]]:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 1: Implement ask_question
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_question(vector_store, llm, question: str) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate an answer.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    if not question or not question.strip():
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    cleaned_question = question.strip()

    # 1. Retrieve top 3 most relevant document chunks
    docs = vector_store.similarity_search(cleaned_question, k=3)
    sources = [doc.page_content for doc in docs]

    # 2. Combine chunk text into context string
    context = "\n\n".join(sources)

    # 3. Format prompt template
    prompt = PROMPT_TEMPLATE.format(context=context, question=cleaned_question)

    # 4. Invoke LLM and extract text
    result = llm(prompt)

    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "generated_text" in result[0]:
        answer = result[0]["generated_text"].strip()
    elif isinstance(result, str):
        answer = result.strip()
    else:
        answer = str(result).strip()

    if not answer:
        answer = "I don't have enough information to answer that."

    return {
        "answer": answer,
        "sources": sources,
    }


def print_formatted_result(question: str, result: Dict[str, Any]) -> None:
    """Helper to display retrieved sources and answer with high-quality formatting."""
    if HAS_RICH and console:
        console.print(f"\n[bold yellow]❓ Question:[/bold yellow] [bold white]{question}[/bold white]\n")
        
        sources_text = ""
        for idx, source in enumerate(result["sources"], 1):
            clean_snippet = source.replace("\n", " ").strip()
            sources_text += f"[bold cyan]{idx}.[/bold cyan] {clean_snippet}\n\n"
        
        console.print(Panel(sources_text.strip(), title="📄 Retrieved Sources (Top Chunks)", border_style="cyan", expand=False))
        console.print(Panel(f"[bold green]{result['answer']}[/bold green]", title="💬 Answer", border_style="green", expand=False))
        console.print()
    else:
        print(f"\n❓ Question: {question}")
        print("\n📄 Sources:")
        for idx, source in enumerate(result["sources"], 1):
            formatted_source = source.replace("\n", " ")
            print(f"  {idx}. {formatted_source}")
        print(f"\n💬 Answer: {result['answer']}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TODO 2: Complete the interactive loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    """Interactive Q&A loop."""
    parser = argparse.ArgumentParser(description="Marketing Agency Interactive Q&A RAG Chatbot")
    parser.add_argument("--query", "-q", type=str, help="Run a single question and exit")
    parser.add_argument("--json", action="store_true", help="Output result as JSON object")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist at {data_dir}", file=sys.stderr)
        sys.exit(1)

    if HAS_RICH and console and not args.json:
        console.print(Panel.fit(
            "[bold magenta]🤖 Marketing Agency RAG Q&A Assistant[/bold magenta]\n"
            "[dim]Powered by LangChain (v0.3), FAISS, & Local Flan-T5 Model[/dim]\n"
            "[italic blue]Everything runs 100% locally on CPU without API keys![/italic blue]",
            border_style="magenta"
        ))

    if HAS_RICH and console and not args.json:
        with console.status("[bold green]Building knowledge base & loading Flan-T5 LLM...", spinner="dots"):
            vector_store = build_knowledge_base(data_dir)
            llm = get_llm()
    else:
        vector_store = build_knowledge_base(data_dir)
        llm = get_llm()

    # Single Query Mode (--query)
    if args.query:
        result = ask_question(vector_store, llm, args.query)
        if args.json:
            import json
            print(json.dumps(result, indent=2))
        else:
            print_formatted_result(args.query, result)
        return

    # Interactive CLI Mode
    if HAS_RICH and console:
        console.print("[bold green]Ready! Ask any question about pricing, services, or process.[/bold green]")
        console.print("[dim]Type 'quit' or 'exit' to terminate.[/dim]\n")
    else:
        print("\n🤖 Marketing Agency Q&A Assistant ready! Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            if HAS_RICH and console:
                query = console.input("[bold cyan]> [/bold cyan]").strip()
            else:
                query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = ask_question(vector_store, llm, query)
        print_formatted_result(query, result)


if __name__ == "__main__":
    main()