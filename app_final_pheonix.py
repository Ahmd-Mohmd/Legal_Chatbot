# -*- coding: utf-8 -*-
"""
Egyptian Legal AI Assistant — Multi-Law RAG Pipeline + Phoenix Observability
=============================================================================
Same architecture as app_final_updated.py with OpenTelemetry tracing for
Arize Phoenix. Traces are exported to a local Phoenix server (OTLP).

Run Phoenix first:  python -m phoenix.server.main serve
Then run this app:  streamlit run app_final_pheonix.py
Phoenix UI:         http://localhost:6006
"""

import os
import sys
import json
import re
import time
import shutil
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

# Suppress progress bars and noisy logs
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# PHOENIX / OPENTELEMETRY TRACING SETUP
# ═══════════════════════════════════════════════════════════════════
PHOENIX_AVAILABLE = False
_phoenix_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    PHOENIX_AVAILABLE = True
except Exception:
    pass


def setup_phoenix_tracing():
    """Configure OTLP tracing for Phoenix."""
    if not PHOENIX_AVAILABLE:
        print("⚠️  Phoenix tracing unavailable (missing opentelemetry packages)")
        return None
    service_name = os.getenv("PHOENIX_SERVICE_NAME", "legal-assistant")
    endpoint = os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    print(f"✅ Phoenix tracing enabled → {endpoint}")
    return trace.get_tracer(service_name)


_phoenix_tracer = setup_phoenix_tracing()


class PhoenixSpan:
    """Context manager for creating OpenTelemetry spans with proper hierarchy."""

    def __init__(self, name: str, attributes: Optional[dict] = None, kind: str = "INTERNAL"):
        self.name = name
        self.attributes = attributes or {}
        self.kind = kind
        self._span_context = None
        self._span = None
        self._t0 = None

    def __enter__(self):
        if _phoenix_tracer:
            from opentelemetry.trace import SpanKind
            self._t0 = time.time()
            kind_map = {"CLIENT": SpanKind.CLIENT, "SERVER": SpanKind.SERVER, "INTERNAL": SpanKind.INTERNAL}
            self._span_context = _phoenix_tracer.start_as_current_span(self.name, kind=kind_map.get(self.kind, SpanKind.INTERNAL))
            self._span = self._span_context.__enter__()
            for k, v in self.attributes.items():
                try:
                    self._span.set_attribute(k, v)
                except Exception:
                    pass
        return self

    def set_attr(self, key: str, value):
        if self._span:
            try:
                self._span.set_attribute(key, value)
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb):
        if self._span_context:
            try:
                if exc_type:
                    self._span.record_exception(exc)
                    from opentelemetry.trace import Status, StatusCode
                    self._span.set_status(Status(StatusCode.ERROR, str(exc)))
                else:
                    if self._t0:
                        self._span.set_attribute("duration_ms", round((time.time() - self._t0) * 1000, 2))
                    from opentelemetry.trace import Status, StatusCode
                    self._span.set_status(Status(StatusCode.OK))
                self._span_context.__exit__(exc_type, exc, tb)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════
# LANGCHAIN & ML IMPORTS
# ═══════════════════════════════════════════════════════════════════
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
import numpy as np
import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION (identical to app_final_updated.py)
# ═══════════════════════════════════════════════════════════════════
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "Omartificial-Intelligence-Space/GATE-AraBert-v1")

SEMANTIC_K: int = 10
BM25_K: int = 10
METADATA_K: int = 10
RRF_K: int = 60
RRF_TOP_K: int = 12
BETA_SEMANTIC: float = 0.50
BETA_BM25: float = 0.30
BETA_METADATA: float = 0.20
RERANKER_TOP_N: int = 5

LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.2
LLM_TOP_P: float = 0.80
LLM_MAX_RETRIES: int = 3
LLM_TIMEOUT: int = 120
CHAT_HISTORY_TURNS: int = 3

# ═══════════════════════════════════════════════════════════════════
# LOGGING & PATHS
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RERANKER_DIR = os.path.join(BASE_DIR, "reranker")
_model_tag = EMBEDDING_MODEL.split("/")[-1].lower().replace("-", "_")
CHROMA_DIR = os.path.join(BASE_DIR, f"chroma_db_{_model_tag}")
EMBEDDING_CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")

# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="المساعد القانوني", page_icon="⚖️")
st.markdown("""
<style>
    .stApp { direction: rtl; text-align: right; }
    .stTextInput input { direction: rtl; text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }
    .stMarkdown p { margin: 0.5em 0 !important; line-height: 1.6; word-spacing: 0.1em; }
    p, div, span, label { unicode-bidi: embed; direction: inherit; white-space: normal; word-wrap: break-word; }
    * { direction: rtl !important; }
    .stMarkdown pre { direction: rtl; text-align: right; white-space: pre-wrap; word-wrap: break-word; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
st.title("⚖️ المساعد القانوني الذكي (مع Phoenix)")

# ═══════════════════════════════════════════════════════════════════
# UTILITY HELPERS (same as app_final_updated.py)
# ═══════════════════════════════════════════════════════════════════

def convert_to_eastern_arabic(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


_ARABIC_STOPWORDS: Set[str] = {
    "في", "من", "على", "إلى", "عن", "أن", "هذا", "هذه", "التي", "الذي",
    "ما", "لا", "أو", "و", "كل", "ذلك", "بين", "كان", "قد", "هو", "هي",
    "لم", "بل", "ثم", "إذا", "حتى", "لكن", "منه", "فيه", "عند", "له",
    "بها", "لها", "منها", "فيها", "التى", "الذى", "ولا", "وفى", "كما",
    "تلك", "هنا", "أي", "دون", "ليس", "إلا", "أما", "مع", "عليه",
}


def arabic_tokenize(text: str) -> List[str]:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _ARABIC_STOPWORDS and len(t) > 1]


def format_chat_history(messages: list, max_turns: int = CHAT_HISTORY_TURNS) -> List:
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "user":
            user_msg = messages[i]["content"]
            ai_msg = ""
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                ai_msg = messages[i + 1]["content"]
                i += 2
            else:
                i += 1
            pairs.append((user_msg, ai_msg))
        else:
            i += 1
    history: List = []
    for user_text, ai_text in pairs[-max_turns:]:
        history.append(HumanMessage(content=user_text))
        if ai_text:
            history.append(AIMessage(content=ai_text))
    return history


# ═══════════════════════════════════════════════════════════════════
# PROMPT (same as app_final_updated.py)
# ═══════════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTIONS: str = """\
<role>
أنت "المساعد القانوني الذكي"، مستشار قانوني متخصص في القوانين المصرية التالية:
• الدستور المصري
• القانون المدني المصري
• قانون العمل المصري
• قانون الأحوال الشخصية المصري
• قانون مكافحة جرائم تقنية المعلومات
• قانون الإجراءات الجنائية المصري

مهمتك: الإجابة بدقة استناداً إلى السياق التشريعي المرفق أدناه.
</role>

<chat_history_instruction>
إذا وُجد سجل محادثة سابق، استخدمه لفهم أسئلة المتابعة والسياق.
لكن دائماً أعطِ الأولوية للسياق التشريعي المسترجع عند الإجابة.
لا تكرر إجابات سابقة بالكامل — أشر إليها باختصار إن لزم.
</chat_history_instruction>

<decision_logic>
حلّل سؤال المستخدم ثم اتبع أول حالة ينطبق شرطها:

━━━ الحالة ١ — الإجابة موجودة في السياق ━━━
• أجب من السياق مباشرةً.
• وثّق بذكر اسم القانون ورقم المادة.
• لا تُضف معلومات من خارج السياق.

━━━ الحالة ٢ — السياق يغطي الموضوع جزئياً ━━━
• اذكر ما تنص عليه المواد المتاحة (مع التوثيق).
• أضف «ملاحظة عملية:» قبل أي إضافة خارجية.

━━━ الحالة ٣ — لا يوجد سياق + سؤال إجرائي ━━━
• ابدأ بـ: «بناءً على الإجراءات القانونية المتعارف عليها في مصر:»
• خطوات مرقمة بدون أرقام مواد.
• أنهِ بـ «يُنصح بمراجعة محامٍ متخصص.»

━━━ الحالة ٤ — لا يوجد سياق + نص قانوني ━━━
• «عذراً، لم يرد ذكر لهذا الموضوع في النصوص المتاحة حالياً.»

━━━ الحالة ٥ — محادثة ودية ━━━
• رد بتحية + «أنا مستشارك القانوني الذكي — اسألني عن أي موضوع في القوانين المصرية.»

━━━ الحالة ٦ — خارج نطاق القانون ━━━
• «تخصصي هو القوانين المصرية فقط.»
</decision_logic>

<quality_rules>
• الدقة أولاً. لا تخترع مراجع. إيجاز مع شمول. نقاط مرقمة عند الحاجة.
</quality_rules>

<formatting_rules>
• ادخل في صلب الموضوع. فقرات قصيرة. لا تكرار. عربية فصحى مبسطة.
</formatting_rules>
"""

# ═══════════════════════════════════════════════════════════════════
# RAG PIPELINE (with Phoenix spans)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def initialize_rag_pipeline():
    _init_t0 = time.time()
    print("🔄 Initialising system…")

    # ── 1. LOAD DATA ─────────────────────────────────────────────
    print("📥 Loading legal data…")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    def _load_json_folder(folder_path: str) -> List[dict]:
        all_items: List[dict] = []
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith(".json"):
                continue
            fpath = os.path.join(folder_path, filename)
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            wrapper_law_name = ""
            if isinstance(obj, list):
                articles: List[dict] = []
                for entry in obj:
                    if isinstance(entry, dict) and "data" in entry and isinstance(entry["data"], list):
                        wrapper_law_name = entry.get("law_name", "")
                        for art in entry["data"]:
                            art.setdefault("_law_name", wrapper_law_name)
                        articles.extend(entry["data"])
                    elif isinstance(entry, dict) and "articles" in entry and isinstance(entry["articles"], list):
                        wrapper_law_name = entry.get("law_name", "")
                        for art in entry["articles"]:
                            art.setdefault("_law_name", wrapper_law_name)
                        articles.extend(entry["articles"])
                    elif isinstance(entry, dict):
                        if not entry.get("_law_name"):
                            aid = entry.get("article_id", "")
                            entry["_law_name"] = "الدستور المصري" if "CONST" in aid.upper() else ""
                        articles.append(entry)
                all_items.extend(articles)
            elif isinstance(obj, dict):
                wrapper_law_name = obj.get("law_name", "")
                data_key = "data" if "data" in obj else ("articles" if "articles" in obj else None)
                if data_key:
                    for art in obj[data_key]:
                        art.setdefault("_law_name", wrapper_law_name)
                    all_items.extend(obj[data_key])
                else:
                    obj.setdefault("_law_name", wrapper_law_name)
                    all_items.append(obj)
        return all_items

    with PhoenixSpan("data_loading", {"data_dir": DATA_DIR}) as dl_span:
        raw_data = _load_json_folder(DATA_DIR)
        dl_span.set_attr("raw_articles", len(raw_data))

    with PhoenixSpan("data_deduplication", {"raw_count": len(raw_data)}) as dedup_span:
        unique: Dict[str, dict] = {}
        for item in raw_data:
            key = str(item.get("article_id") or item.get("article_number") or hashlib.md5(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            unique[key] = item
        data = list(unique.values())
        dedup_span.set_attr("unique_count", len(data))
        dedup_span.set_attr("duplicates_removed", len(raw_data) - len(data))

    docs: List[Document] = []
    for item in data:
        article_number = item.get("article_number")
        original_text = item.get("original_text")
        simplified_summary = item.get("simplified_summary")
        if not article_number or not original_text or not simplified_summary:
            continue
        law_name = item.get("law_name") or item.get("_law_name", "")
        part_bab = item.get("part (Bab)", "")
        chapter_fasl = item.get("chapter (Fasl)", "")
        page_content = (
            f"القانون: {law_name}\nرقم المادة: {article_number}\n"
            f"الباب: {part_bab}\nالفصل: {chapter_fasl}\n"
            f"النص الأصلي: {original_text}\nالشرح المبسط: {simplified_summary}"
        )
        metadata = {
            "article_id": item.get("article_id") or str(article_number),
            "article_number": str(article_number),
            "law_name": law_name,
            "legal_nature": item.get("legal_nature", ""),
            "keywords": ", ".join(item.get("keywords", []) or []),
            "part": part_bab,
            "chapter": chapter_fasl,
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    print(f"✅ Loaded {len(docs)} legal articles")

    # ── 2. EMBEDDINGS (cached locally after first download) ─────
    print(f"📐 Loading embedding model: {EMBEDDING_MODEL}")
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    with PhoenixSpan("embedding_model_load", {
        "model_name": EMBEDDING_MODEL,
        "cache_dir": EMBEDDING_CACHE_DIR,
        "cache_hit": os.path.exists(os.path.join(EMBEDDING_CACHE_DIR, _model_tag)),
    }) as emb_span:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=EMBEDDING_CACHE_DIR,
        )
        emb_span.set_attr("status", "loaded")
    print(f"✅ Embeddings ready (cache: {EMBEDDING_CACHE_DIR})")

    # ── 3. VECTOR STORE ──────────────────────────────────────────
    db_exists = os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)
    with PhoenixSpan("vectorstore_init", {
        "chroma_dir": CHROMA_DIR,
        "cache_exists": bool(db_exists),
    }) as vs_span:
        if db_exists:
            print("📦 Loading persisted Chroma DB…")
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            stored_count = vectorstore._collection.count()
            if stored_count == 0 or abs(stored_count - len(docs)) > 5:
                print(f"⚠️  Count mismatch. Rebuilding…")
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                db_exists = False
                vs_span.set_attr("action", "rebuild_triggered")
            else:
                print(f"✅ Chroma DB loaded ({stored_count} vectors)")
                vs_span.set_attr("action", "loaded_from_cache")
                vs_span.set_attr("vector_count", stored_count)
        if not db_exists:
            print("🧱 Building Chroma DB…")
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
            print(f"✅ Chroma DB built ({len(docs)} vectors)")
            vs_span.set_attr("action", "built_from_scratch")
            vs_span.set_attr("vector_count", len(docs))

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": SEMANTIC_K})

    # ── 4. BM25 RETRIEVER ────────────────────────────────────────
    class BM25Retriever(BaseRetriever):
        corpus_docs: List[Document]
        bm25: Optional[BM25Okapi] = None
        tokenized_corpus: Optional[list] = None
        k: int = BM25_K
        class Config:
            arbitrary_types_allowed = True
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tokenized_corpus = [arabic_tokenize(d.page_content) for d in self.corpus_docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            tokens = arabic_tokenize(query)
            if not tokens:
                return []
            scores = self.bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:self.k]
            return [self.corpus_docs[i] for i in top_idx if scores[i] > 0]
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    with PhoenixSpan("bm25_index_build", {"corpus_size": len(docs), "k": BM25_K}) as bm25_span:
        bm25_retriever = BM25Retriever(corpus_docs=docs, k=BM25_K)
        bm25_span.set_attr("vocab_size", len(set(t for toks in bm25_retriever.tokenized_corpus for t in toks)))
    print("✅ BM25 retriever ready")

    # ── 5. METADATA RETRIEVER ────────────────────────────────────
    class MetadataFilterRetriever(BaseRetriever):
        corpus_docs: List[Document]
        keyword_index: Optional[Dict[str, Set[int]]] = None
        law_name_index: Optional[Dict[str, Set[int]]] = None
        k: int = METADATA_K
        class Config:
            arbitrary_types_allowed = True
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.keyword_index = defaultdict(set)
            self.law_name_index = defaultdict(set)
            for idx, doc in enumerate(self.corpus_docs):
                kw_text = " ".join([
                    doc.metadata.get("keywords", ""),
                    doc.metadata.get("legal_nature", ""),
                    doc.metadata.get("part", ""),
                    doc.metadata.get("chapter", ""),
                ])
                for tok in arabic_tokenize(kw_text):
                    self.keyword_index[tok].add(idx)
                for tok in arabic_tokenize(doc.metadata.get("law_name", "")):
                    self.law_name_index[tok].add(idx)
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            tokens = arabic_tokenize(query)
            if not tokens:
                return []
            scores: Dict[int, float] = defaultdict(float)
            for tok in tokens:
                for idx in self.keyword_index.get(tok, set()):
                    scores[idx] += 3.0
                for idx in self.law_name_index.get(tok, set()):
                    scores[idx] += 4.0
            if not scores:
                return []
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.k]
            return [self.corpus_docs[idx] for idx, _ in top]
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    with PhoenixSpan("metadata_index_build", {"corpus_size": len(docs), "k": METADATA_K}) as meta_span:
        metadata_retriever = MetadataFilterRetriever(corpus_docs=docs, k=METADATA_K)
        meta_span.set_attr("keyword_terms", len(metadata_retriever.keyword_index))
        meta_span.set_attr("law_name_terms", len(metadata_retriever.law_name_index))
    print("✅ Metadata retriever ready")

    # ── 6. HYBRID RRF (with Phoenix span) ────────────────────────
    class HybridRRFRetriever(BaseRetriever):
        semantic_retriever: BaseRetriever
        bm25_retriever: BM25Retriever
        metadata_retriever: MetadataFilterRetriever
        beta_semantic: float = BETA_SEMANTIC
        beta_keyword: float = BETA_BM25
        beta_metadata: float = BETA_METADATA
        k: int = RRF_K
        top_k: int = RRF_TOP_K
        class Config:
            arbitrary_types_allowed = True
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            with PhoenixSpan("hybrid_retrieval", {
                "query": query[:200],
                "beta_semantic": self.beta_semantic,
                "beta_keyword": self.beta_keyword,
                "beta_metadata": self.beta_metadata,
            }) as span:
                # ── Run 3 retrievers in parallel with individual spans ──
                # Capture current OTel context so threads inherit the parent span
                if PHOENIX_AVAILABLE:
                    from opentelemetry import context as otel_context
                    _parent_ctx = otel_context.get_current()
                else:
                    _parent_ctx = None

                def _run_semantic(q):
                    if _parent_ctx:
                        token = otel_context.attach(_parent_ctx)
                    try:
                        with PhoenixSpan("semantic_retrieval", {"query": q[:200], "k": SEMANTIC_K}) as s:
                            docs = self.semantic_retriever.invoke(q)
                            s.set_attr("result_count", len(docs))
                            if docs:
                                s.set_attr("top_article", docs[0].metadata.get("article_number", "?"))
                            return docs
                    finally:
                        if _parent_ctx:
                            otel_context.detach(token)

                def _run_bm25(q):
                    if _parent_ctx:
                        token = otel_context.attach(_parent_ctx)
                    try:
                        with PhoenixSpan("bm25_retrieval", {"query": q[:200], "k": BM25_K}) as s:
                            tokens = arabic_tokenize(q)
                            s.set_attr("query_tokens", len(tokens))
                            docs = self.bm25_retriever.invoke(q)
                            s.set_attr("result_count", len(docs))
                            if docs:
                                s.set_attr("top_article", docs[0].metadata.get("article_number", "?"))
                            return docs
                    finally:
                        if _parent_ctx:
                            otel_context.detach(token)

                def _run_metadata(q):
                    if _parent_ctx:
                        token = otel_context.attach(_parent_ctx)
                    try:
                        with PhoenixSpan("metadata_retrieval", {"query": q[:200], "k": METADATA_K}) as s:
                            docs = self.metadata_retriever.invoke(q)
                            s.set_attr("result_count", len(docs))
                            if docs:
                                s.set_attr("top_article", docs[0].metadata.get("article_number", "?"))
                            return docs
                    finally:
                        if _parent_ctx:
                            otel_context.detach(token)

                with ThreadPoolExecutor(max_workers=3) as pool:
                    f_sem = pool.submit(_run_semantic, query)
                    f_bm25 = pool.submit(_run_bm25, query)
                    f_meta = pool.submit(_run_metadata, query)
                    semantic_docs = f_sem.result(timeout=30)
                    bm25_docs = f_bm25.result(timeout=30)
                    meta_docs = f_meta.result(timeout=30)

                span.set_attr("semantic_count", len(semantic_docs))
                span.set_attr("bm25_count", len(bm25_docs))
                span.set_attr("metadata_count", len(meta_docs))
                span.set_attr("total_candidates", len(semantic_docs) + len(bm25_docs) + len(meta_docs))

                # ── RRF Fusion with its own span ──
                with PhoenixSpan("rrf_fusion", {
                    "rrf_k": self.k,
                    "top_k": self.top_k,
                    "input_docs": len(semantic_docs) + len(bm25_docs) + len(meta_docs),
                }) as fusion_span:
                    rrf_scores: Dict[str, float] = {}
                    all_docs: Dict[str, Document] = {}
                    for weight, doc_list in [
                        (self.beta_semantic, semantic_docs),
                        (self.beta_keyword, bm25_docs),
                        (self.beta_metadata, meta_docs),
                    ]:
                        for rank, doc in enumerate(doc_list, start=1):
                            did = doc.metadata.get("article_id") or doc.metadata.get("article_number") or str(hash(doc.page_content))
                            rrf_scores[did] = rrf_scores.get(did, 0.0) + weight / (self.k + rank)
                            all_docs.setdefault(did, doc)

                    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                    result = [all_docs[did] for did, _ in sorted_ids[:self.top_k] if did in all_docs]
                    fusion_span.set_attr("unique_docs", len(all_docs))
                    fusion_span.set_attr("output_count", len(result))
                    if sorted_ids:
                        fusion_span.set_attr("top_score", round(sorted_ids[0][1], 6))
                        fusion_span.set_attr("bottom_score", round(sorted_ids[min(len(sorted_ids)-1, self.top_k-1)][1], 6))
                    # Log overlap between retrievers
                    sem_ids = {d.metadata.get("article_id", "") for d in semantic_docs}
                    bm25_ids = {d.metadata.get("article_id", "") for d in bm25_docs}
                    meta_ids = {d.metadata.get("article_id", "") for d in meta_docs}
                    fusion_span.set_attr("overlap_sem_bm25", len(sem_ids & bm25_ids))
                    fusion_span.set_attr("overlap_sem_meta", len(sem_ids & meta_ids))
                    fusion_span.set_attr("overlap_bm25_meta", len(bm25_ids & meta_ids))
                    fusion_span.set_attr("overlap_all_three", len(sem_ids & bm25_ids & meta_ids))

                span.set_attr("fused_count", len(result))
                if result:
                    span.set_attr("top_articles", ", ".join(d.metadata.get("article_number", "?") for d in result[:5]))
                    # Log which laws are represented
                    laws = set(d.metadata.get("law_name", "?") for d in result)
                    span.set_attr("laws_represented", ", ".join(laws))
                return result
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    hybrid_retriever = HybridRRFRetriever(
        semantic_retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=BETA_SEMANTIC, beta_keyword=BETA_BM25, beta_metadata=BETA_METADATA,
        k=RRF_K, top_k=RRF_TOP_K,
    )
    print(f"✅ Hybrid RRF retriever (β: {BETA_SEMANTIC}/{BETA_BM25}/{BETA_METADATA})")

    # ── 7. RERANKER (with Phoenix span) ──────────────────────────
    print("🔃 Loading reranker…")
    if not os.path.exists(RERANKER_DIR):
        raise FileNotFoundError(f"Reranker not found: {RERANKER_DIR}")

    with PhoenixSpan("reranker_model_load", {"model_dir": RERANKER_DIR, "top_n": RERANKER_TOP_N}) as rr_span:
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_DIR)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=RERANKER_TOP_N)
        _base_compression = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid_retriever)
        rr_span.set_attr("status", "loaded")

    class InstrumentedCompressionRetriever(BaseRetriever):
        base_retriever: ContextualCompressionRetriever
        class Config:
            arbitrary_types_allowed = True
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            with PhoenixSpan("reranker_compression", {
                "query": query[:200],
                "top_n": RERANKER_TOP_N,
                "model": "ARM-V1",
            }) as span:
                docs = self.base_retriever.invoke(query)
                span.set_attr("input_count", RRF_TOP_K)
                span.set_attr("output_count", len(docs))
                if docs:
                    span.set_attr("articles", ", ".join(d.metadata.get("article_number", "?") for d in docs))
                    span.set_attr("laws", ", ".join(set(d.metadata.get("law_name", "?") for d in docs)))
                    # Estimate relevance by checking how many unique laws appear
                    span.set_attr("unique_laws_count", len(set(d.metadata.get("law_name", "") for d in docs)))
                return docs
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    compression_retriever = InstrumentedCompressionRetriever(base_retriever=_base_compression)
    print(f"✅ Reranker ready (top_n={RERANKER_TOP_N})")

    # ── 8. LLM ───────────────────────────────────────────────────
    _groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not _groq_key:
        raise RuntimeError("GROQ_API_KEY not set in environment / .env file")
    llm = ChatGroq(
        groq_api_key=_groq_key,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        model_kwargs={"top_p": LLM_TOP_P},
        max_retries=LLM_MAX_RETRIES,
        request_timeout=LLM_TIMEOUT,
    )

    # ── 9. PROMPT ────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTIONS),
        ("system", "السياق التشريعي المتاح (المصدر الأساسي):\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "سؤال المستفيد:\n{input}"),
    ])

    # ── 10. CHAIN ────────────────────────────────────────────────
    def _format_context(docs: List[Document]) -> str:
        return "\n\n---\n\n".join(d.page_content for d in docs)

    qa_chain = (
        RunnableParallel({
            "context": (lambda x: x["input"]) | compression_retriever,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
        })
        .assign(
            answer=(
                RunnableLambda(lambda x: {
                    "context": _format_context(x["context"]),
                    "input": x["input"],
                    "chat_history": x.get("chat_history", []),
                })
                | prompt
                | llm
                | StrOutputParser()
            ),
        )
    )

    # ── Log total initialization time ────────────────────────────
    _init_duration = round((time.time() - _init_t0) * 1000, 2)
    with PhoenixSpan("pipeline_initialization_complete", {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "total_docs": len(docs),
        "total_init_ms": _init_duration,
        "chroma_dir": CHROMA_DIR,
        "reranker_top_n": RERANKER_TOP_N,
        "rrf_top_k": RRF_TOP_K,
        "chat_history_turns": CHAT_HISTORY_TURNS,
    }) as final_span:
        final_span.set_attr("status", "ready")
    print(f"✅ System ready! (init: {_init_duration:.0f}ms)")
    return qa_chain


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════
try:
    qa_chain = initialize_rag_pipeline()
except Exception as e:
    st.error(f"خطأ فادح: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(convert_to_eastern_arabic(msg["content"]))

if prompt_input := st.chat_input("اكتب سؤالك القانوني هنا…"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري التحليل القانوني…"):
            try:
                # ════════════════════════════════════════════════════
                # ROOT SPAN — every child span nests under this tree
                # ════════════════════════════════════════════════════
                with PhoenixSpan("chat_request", {
                    "question": prompt_input[:500],
                    "question_length": len(prompt_input),
                    "session_messages": len(st.session_state.messages),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, kind="SERVER") as req_span:

                    # ├─ 1. chat_history_format
                    with PhoenixSpan("chat_history_format", {
                        "raw_messages": len(st.session_state.messages) - 1,
                        "max_turns": CHAT_HISTORY_TURNS,
                    }) as hist_span:
                        history = format_chat_history(st.session_state.messages[:-1])
                        hist_span.set_attr("formatted_messages", len(history))
                        hist_span.set_attr("has_history", len(history) > 0)

                    req_span.set_attr("chat_history_turns", len(history) // 2)

                    # ├─ 2. chain_invoke
                    # │   ├─ hybrid_retrieval  (auto-nested from retriever)
                    # │   │   ├─ semantic_retrieval
                    # │   │   ├─ bm25_retrieval
                    # │   │   ├─ metadata_retrieval
                    # │   │   └─ rrf_fusion
                    # │   ├─ reranker_compression  (auto-nested from retriever)
                    # │   └─ llm_call  (Groq API)
                    with PhoenixSpan("chain_invoke", {
                        "input": prompt_input[:300],
                        "history_messages": len(history),
                    }) as chain_span:
                        result = qa_chain.invoke({"input": prompt_input, "chat_history": history})

                        response_text = result.get("answer", "") if isinstance(result, dict) else str(result)
                        source_docs = result.get("context", []) if isinstance(result, dict) else []

                        chain_span.set_attr("answer_length", len(response_text))
                        chain_span.set_attr("context_docs", len(source_docs))

                    # ├─ 3. retrieval_quality
                    with PhoenixSpan("retrieval_quality", {
                        "context_count": len(source_docs),
                    }) as qual_span:
                        if source_docs:
                            articles = [d.metadata.get("article_number", "?") for d in source_docs]
                            laws = [d.metadata.get("law_name", "?") for d in source_docs]
                            natures = [d.metadata.get("legal_nature", "?") for d in source_docs]
                            qual_span.set_attr("articles", ", ".join(articles))
                            qual_span.set_attr("laws", ", ".join(set(laws)))
                            qual_span.set_attr("legal_natures", ", ".join(set(n for n in natures if n)))
                            qual_span.set_attr("unique_laws", len(set(laws)))
                            qual_span.set_attr("unique_articles", len(set(articles)))
                            total_ctx_chars = sum(len(d.page_content) for d in source_docs)
                            qual_span.set_attr("total_context_chars", total_ctx_chars)
                        else:
                            qual_span.set_attr("status", "no_context_found")

                    # ├─ 4. llm_generation (post-hoc analysis)
                    with PhoenixSpan("llm_generation", {
                        "model": LLM_MODEL,
                        "temperature": LLM_TEMPERATURE,
                        "top_p": LLM_TOP_P,
                    }, kind="CLIENT") as llm_span:
                        llm_span.set_attr("response_length", len(response_text))
                        llm_span.set_attr("response_preview", response_text[:500])
                        citation_count = len(re.findall(r"(?:مادة|المادة)\s*\d+", response_text))
                        llm_span.set_attr("citation_count", citation_count)
                        llm_span.set_attr("has_citations", citation_count > 0)

                    # ├─ 5. response_rendering
                    with PhoenixSpan("response_rendering", {
                        "answer_length": len(response_text),
                        "source_docs": len(source_docs),
                    }) as render_span:
                        st.markdown(convert_to_eastern_arabic(response_text))

                        if source_docs:
                            seen: Set[str] = set()
                            unique_docs: List[Document] = []
                            for doc in source_docs:
                                art_num = str(doc.metadata.get("article_number", "")).strip()
                                if art_num and art_num not in seen:
                                    seen.add(art_num)
                                    unique_docs.append(doc)

                            render_span.set_attr("unique_source_count", len(unique_docs))
                            st.markdown("---")
                            if unique_docs:
                                with st.expander(f"📚 المصادر ({len(unique_docs)} مادة)"):
                                    for doc in unique_docs:
                                        art_num = str(doc.metadata.get("article_number", "")).strip()
                                        law = doc.metadata.get("law_name", "")
                                        nature = doc.metadata.get("legal_nature", "")
                                        header = f"**المادة رقم {convert_to_eastern_arabic(art_num)}**"
                                        if law:
                                            header += f" — {law}"
                                        st.markdown(header)
                                        if nature:
                                            st.markdown(f"*{nature}*")
                                        for line in doc.page_content.strip().split("\n"):
                                            line = line.strip()
                                            if line:
                                                st.markdown(convert_to_eastern_arabic(line))
                                        st.markdown("---")
                            else:
                                st.info("📌 لم يتم العثور على مصادر")
                        else:
                            st.info("📌 لم يتم العثور على مصادر")

                    # └─ Summary on root span
                    req_span.set_attr("answer_length", len(response_text))
                    req_span.set_attr("context_count", len(source_docs))
                    req_span.set_attr("response_preview", response_text[:300])
                    req_span.set_attr("status", "success")

                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                if _phoenix_tracer:
                    with PhoenixSpan("chat_request_error", {
                        "question": prompt_input[:300],
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:500],
                    }) as err_span:
                        pass
                st.error(f"حدث خطأ: {e}")
