"""
Ingestion Service - Main
------------------------
Orchestrates:
- config
- file-based ingestion
- live API ingestion
- normalization to text
- preprocessing (clean + chunk)
- indexing (dummy embeddings + stubs)
"""

from config import get_settings

from ingestion.File_sources.pdf_ingestor import ingest_pdfs
from ingestion.File_sources.csv_ingestor import ingest_csvs
from ingestion.File_sources.json_ingestor import ingest_json

from ingestion.live_sources.market_api_client import fetch_market_data
from ingestion.live_sources.news_api_client import fetch_news_for_ticker
from ingestion.live_sources.sec_api_client import fetch_sec_filings

from ingestion.normalize.text_from_market import build_market_sentences
from ingestion.normalize.text_from_news import build_news_sentences
from ingestion.normalize.text_from_sec import build_sec_sentences

from preprocessing.text_cleaning import clean_text
from preprocessing.chunking import simple_chunk

from indexing.embeddings_adapter import dummy_embed_texts
from indexing.vector_indexer import index_vectors
from indexing.graph_indexer import update_graph_for_company


def run_ingestion():
    settings = get_settings()

    print("[ingestion-service] Starting ingestion run...")
    print(f"[ingestion-service] Data directory: {settings.data_dir}")

    # ---------- File-based ingestion ----------
    if settings.enable_pdf_ingestion:
        pdf_dir = settings.data_dir / "raw" / "sec"
        pdf_texts = ingest_pdfs(pdf_dir)
    else:
        pdf_texts = []
        print("[ingestion-service] PDF ingestion disabled.")

    if settings.enable_csv_ingestion:
        csv_dir = settings.data_dir / "raw" / "uploads"
        csv_rows = ingest_csvs(csv_dir)
        if csv_rows:
            print(f"[ingestion-service] Example CSV row: {csv_rows[0]}")
        else:
            print("[ingestion-service] No CSV rows ingested.")
    else:
        csv_rows = []
        print("[ingestion-service] CSV ingestion disabled.")

    if settings.enable_json_ingestion:
        json_dir = settings.data_dir / "raw" / "json"
        json_items = ingest_json(json_dir)
        print(f"[ingestion-service] JSON objects ingested: {len(json_items)}")
    else:
        json_items = []
        print("[ingestion-service] JSON ingestion disabled.")

    # ---------- Live sources ingestion ----------
    market_sentences_all = []
    news_sentences_all = []
    sec_sentences_all = []

    if settings.enable_live_market:
        print("[ingestion-service] Fetching real-time market data for NVDA and AAPL...")
        nvda_data = fetch_market_data("NVDA", period="1mo", interval="1d")
        aapl_data = fetch_market_data("AAPL", period="1mo", interval="1d")

        print("[ingestion-service] NVDA recent prices (tail 3):")
        for row in nvda_data["history"][-3:]:
            print(f"  {row}")
        print("[ingestion-service] AAPL recent prices (tail 3):")
        for row in aapl_data["history"][-3:]:
            print(f"  {row}")

        market_sentences_all.extend(build_market_sentences(nvda_data))
        market_sentences_all.extend(build_market_sentences(aapl_data))
    else:
        print("[ingestion-service] Live market ingestion disabled.")

    if settings.enable_live_news:
        print("[ingestion-service] Fetching news for NVDA and AAPL...")
        nvda_news = fetch_news_for_ticker("NVDA", limit=5)
        aapl_news = fetch_news_for_ticker("AAPL", limit=5)
        news_sentences_all.extend(build_news_sentences(nvda_news))
        news_sentences_all.extend(build_news_sentences(aapl_news))
    else:
        print("[ingestion-service] Live news ingestion disabled.")

    # SEC API stub
    sec_filings_nvda = fetch_sec_filings("NVDA", limit=1)
    sec_sentences_all.extend(build_sec_sentences(sec_filings_nvda))

    # ---------- Combine all texts ----------
    all_texts = []

    all_texts.extend(pdf_texts)
    all_texts.extend(market_sentences_all)
    all_texts.extend(news_sentences_all)
    all_texts.extend(sec_sentences_all)

    if not all_texts:
        all_texts.append("This is a SAMPLE raw text from a PDF or CSV or JSON.")

    print(f"[ingestion-service] Total text items before preprocessing: {len(all_texts)}")

    # ---------- Preprocessing + embeddings ----------
    cleaned_chunks = []
    for raw in all_texts:
        cleaned = clean_text(raw)
        chunks = simple_chunk(cleaned, max_chars=200)
        cleaned_chunks.extend(chunks)

    print(f"[preprocessing] Total chunks: {len(cleaned_chunks)}")

    embeddings = dummy_embed_texts(cleaned_chunks)
    print(f"[embedding] Created {len(embeddings)} embedding vectors.")

    for i, emb in enumerate(embeddings[:3], start=1):
        print(f"  - Embedding {i}: text_len={len(emb.text)}, vector={emb.vector}")

    # ---------- Indexing (stubs) ----------
    index_vectors(embeddings, index_name="finance-demo-index")
    update_graph_for_company("NVDA", ["TSMC", "ASML", "Apple"])

    print("[ingestion-service] Ingestion complete (stub/partial real).")


def main():
    run_ingestion()


if __name__ == "__main__":
    main()
