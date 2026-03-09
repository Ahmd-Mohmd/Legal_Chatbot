# -*- coding: utf-8 -*-
"""
Egyptian Legal AI Assistant — Multi-Law RAG Pipeline
=====================================================
Architecture:
  1. Data Ingestion  → Load 6 Egyptian law JSON files, de-duplicate articles
  2. Embedding       → HuggingFace Arabic embeddings → ChromaDB vector store
  3. Retrieval       → Hybrid RRF (Semantic + BM25 + Metadata) in parallel
  4. Reranking       → Cross-encoder reranker (ARM-V1, Arabic-tuned)
  5. Generation      → Groq LLM (Llama 3.3 70B) with chat history context
  6. UI              → Streamlit with Arabic RTL support

Supports A/B embedding comparison via EMBEDDING_MODEL config.
Chat history (last 3 turns) is injected into the LLM for follow-up awareness.
"""

import os
import sys
import json
import re
import shutil
import hashlib
from dotenv import load_dotenv
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

# Suppress progress bars and noisy logs
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
warnings.filterwarnings("ignore")

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

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════
# -- Embedding A/B switch --
# Options:
#   "Omartificial-Intelligence-Space/GATE-AraBert-v1"              (current default)
#   "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2" (new candidate)
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "Omartificial-Intelligence-Space/GATE-AraBert-v1",
)

# -- Retrieval parameters --
SEMANTIC_K: int = 10          # top-k for semantic (vector) retriever
BM25_K: int = 10              # top-k for BM25 keyword retriever
METADATA_K: int = 10          # top-k for metadata filter retriever
RRF_K: int = 60               # RRF constant (standard = 60)
RRF_TOP_K: int = 12           # final docs after RRF fusion
BETA_SEMANTIC: float = 0.50   # RRF weight for semantic
BETA_BM25: float = 0.30       # RRF weight for BM25
BETA_METADATA: float = 0.20   # RRF weight for metadata

# -- Reranker --
RERANKER_TOP_N: int = 5       # top-n docs after cross-encoder reranking

# -- LLM parameters --
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.2   # low for legal precision
LLM_TOP_P: float = 0.80        # focused sampling
LLM_MAX_RETRIES: int = 3
LLM_TIMEOUT: int = 120         # seconds

# -- Chat history --
CHAT_HISTORY_TURNS: int = 3    # number of past Q&A pairs to include

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RERANKER_DIR = os.path.join(BASE_DIR, "reranker")

# ChromaDB subfolder per embedding model to avoid rebuild on switch
_model_tag = EMBEDDING_MODEL.split("/")[-1].lower().replace("-", "_")
CHROMA_DIR = os.path.join(BASE_DIR, f"chroma_db_{_model_tag}")

# Local cache for embedding model weights (avoids re-downloading)
EMBEDDING_CACHE_DIR = os.path.join(BASE_DIR, "embedding_cache")

# ═══════════════════════════════════════════════════════════════════
# RUNTIME DETECTION (Streamlit vs. script import)
# ═══════════════════════════════════════════════════════════════════
_IS_STREAMLIT = False
try:
    import streamlit as st
    if hasattr(st, "runtime") and st.runtime.exists():
        _IS_STREAMLIT = True
except Exception:
    pass

if _IS_STREAMLIT:
    import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# UI SETUP (only in Streamlit)
# ═══════════════════════════════════════════════════════════════════
if _IS_STREAMLIT:
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
    st.title("⚖️ المساعد القانوني الذكي")

# ═══════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════

def convert_to_eastern_arabic(text: str) -> str:
    """Convert Western numerals (0-9) to Eastern Arabic numerals."""
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
    """Tokenise Arabic text: strip diacritics, keep Arabic chars, remove stopwords."""
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)       # strip tashkeel
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)         # keep Arabic only
    tokens = text.split()
    return [t for t in tokens if t not in _ARABIC_STOPWORDS and len(t) > 1]


def format_chat_history(messages: list, max_turns: int = CHAT_HISTORY_TURNS) -> List:
    """Extract the last *max_turns* Q&A pairs as LangChain message objects."""
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
# PROMPT TEMPLATE
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
الشرط: توجد مادة أو أكثر في السياق تتناول الموضوع.
• أجب من السياق مباشرةً.
• وثّق بذكر اسم القانون ورقم المادة (مثال: «وفقاً للمادة (٥٢) من قانون العمل…»).
• استخرج ما يجيب السؤال تحديداً — لا تنسخ المادة كاملة.
• لا تُضف معلومات من خارج السياق.

━━━ الحالة ٢ — السياق يغطي الموضوع جزئياً ━━━
• اذكر أولاً ما تنص عليه المواد المتاحة (مع التوثيق).
• أضف توضيحاً عملياً مختصراً مع عبارة «ملاحظة عملية:» قبل أي إضافة.
• لا تخترع أرقام مواد.

━━━ الحالة ٣ — لا يوجد سياق + سؤال إجرائي/عملي ━━━
• ابدأ بـ: «بناءً على الإجراءات القانونية المتعارف عليها في مصر:»
• قدّم خطوات مرقمة مختصرة.
• لا تذكر أرقام مواد.
• أنهِ بـ «يُنصح بمراجعة محامٍ متخصص.»

━━━ الحالة ٤ — لا يوجد سياق + سؤال عن نص قانوني ━━━
• قل: «عذراً، لم يرد ذكر لهذا الموضوع في النصوص المتاحة حالياً.»
• لا تجب من ذاكرتك.

━━━ الحالة ٥ — محادثة ودية ━━━
• رد بتحية لطيفة مقتضبة + «أنا مستشارك القانوني الذكي — اسألني عن أي موضوع في القوانين المصرية.»

━━━ الحالة ٦ — خارج نطاق القانون ━━━
• اعتذر بلطف: «تخصصي هو القوانين المصرية فقط.»
</decision_logic>

<quality_rules>
• الدقة أولاً: التزم بالنص القانوني حرفياً عند وجوده.
• لا تخترع مراجع: لا تنسب معلومة إلى مادة لم ترد في السياق.
• الإيجاز مع الشمول: أجب بقدر ما يحتاج السؤال.
• استخدم نقاطاً (•) أو ترقيماً عند ذكر عدة بنود.
</quality_rules>

<formatting_rules>
• لا تكرر هذه التعليمات في ردك.
• ادخل في صلب الموضوع فوراً.
• فقرات قصيرة مفصولة بسطر فارغ.
• لا تكرر نفس المعلومة أو نفس المادة.
• رتّب المواد ترتيباً منطقياً.
• التزم بالعربية الفصحى المبسطة.
</formatting_rules>
"""

# ═══════════════════════════════════════════════════════════════════
# RAG PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════════════
_pipeline_cache = None


def _initialize_rag_pipeline_impl():
    """Build the full RAG pipeline (data → embed → retrieve → rerank → generate).

    Returns a qa_chain that can be invoked with:
        qa_chain.invoke({"input": query, "chat_history": [HumanMessage, AIMessage, ...]})
    and returns {"context": list[Document], "input": str, "chat_history": list, "answer": str}.
    """
    print("🔄 Initialising system…")

    # ── 1. LOAD DATA ─────────────────────────────────────────────
    print("📥 Loading legal data…")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    def _load_json_folder(folder_path: str) -> List[dict]:
        """Load all JSON files from *folder_path*, propagating law_name to every article."""
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

    raw_data = _load_json_folder(DATA_DIR)

    # De-duplicate by article_id
    unique: Dict[str, dict] = {}
    for item in raw_data:
        key = str(item.get("article_id") or item.get("article_number") or hashlib.md5(json.dumps(item, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
        unique[key] = item
    data = list(unique.values())

    # Build Document objects
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
            f"القانون: {law_name}\n"
            f"رقم المادة: {article_number}\n"
            f"الباب: {part_bab}\n"
            f"الفصل: {chapter_fasl}\n"
            f"النص الأصلي: {original_text}\n"
            f"الشرح المبسط: {simplified_summary}"
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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=EMBEDDING_CACHE_DIR,
    )
    print(f"✅ Embeddings ready (cache: {EMBEDDING_CACHE_DIR})")

    # ── 3. VECTOR STORE (ChromaDB) ───────────────────────────────
    db_exists = os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)
    if db_exists:
        print("📦 Loading persisted Chroma DB…")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        stored_count = vectorstore._collection.count()
        if stored_count == 0 or abs(stored_count - len(docs)) > 5:
            print(f"⚠️  Count mismatch (stored={stored_count}, data={len(docs)}). Rebuilding…")
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            db_exists = False
        else:
            print(f"✅ Chroma DB loaded ({stored_count} vectors)")

    if not db_exists:
        print("🧱 Building Chroma DB (first time for this embedding)…")
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
        print(f"✅ Chroma DB built ({len(docs)} vectors)")

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": SEMANTIC_K})

    # ── 4. BM25 KEYWORD RETRIEVER ────────────────────────────────
    class BM25Retriever(BaseRetriever):
        """BM25-based keyword retriever with Arabic-aware tokenisation."""
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
            top_idx = np.argsort(scores)[::-1][: self.k]
            return [self.corpus_docs[i] for i in top_idx if scores[i] > 0]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    bm25_retriever = BM25Retriever(corpus_docs=docs, k=BM25_K)
    print("✅ BM25 retriever ready (Arabic tokeniser)")

    # ── 5. METADATA FILTER RETRIEVER (inverted index) ────────────
    class MetadataFilterRetriever(BaseRetriever):
        """Fast metadata retriever using a pre-built inverted index."""
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
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.k]
            return [self.corpus_docs[idx] for idx, _ in top]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    metadata_retriever = MetadataFilterRetriever(corpus_docs=docs, k=METADATA_K)
    print("✅ Metadata retriever ready (inverted index)")

    # ── 6. HYBRID RRF RETRIEVER (parallel) ───────────────────────
    class HybridRRFRetriever(BaseRetriever):
        """Reciprocal Rank Fusion over semantic + BM25 + metadata retrievers.
        All sub-retrievers run in parallel via ThreadPoolExecutor."""
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
            with ThreadPoolExecutor(max_workers=3) as pool:
                f_sem = pool.submit(self.semantic_retriever.invoke, query)
                f_bm25 = pool.submit(self.bm25_retriever.invoke, query)
                f_meta = pool.submit(self.metadata_retriever.invoke, query)
                semantic_docs = f_sem.result(timeout=30)
                bm25_docs = f_bm25.result(timeout=30)
                meta_docs = f_meta.result(timeout=30)

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
            return [all_docs[did] for did, _ in sorted_ids[: self.top_k] if did in all_docs]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    hybrid_retriever = HybridRRFRetriever(
        semantic_retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=BETA_SEMANTIC,
        beta_keyword=BETA_BM25,
        beta_metadata=BETA_METADATA,
        k=RRF_K,
        top_k=RRF_TOP_K,
    )
    print(f"✅ Hybrid RRF retriever (β: sem={BETA_SEMANTIC}, bm25={BETA_BM25}, meta={BETA_METADATA})")

    # ── 7. RERANKER ──────────────────────────────────────────────
    print("🔃 Loading reranker…")
    if not os.path.exists(RERANKER_DIR):
        raise FileNotFoundError(f"Reranker not found: {RERANKER_DIR}")

    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_DIR)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=RERANKER_TOP_N)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever,
    )
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

    # ── 9. PROMPT (with chat history) ───────────────────────────
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

    print("✅ System ready!")
    return qa_chain


def initialize_rag_pipeline():
    """Public entry point — cached in Streamlit, module-level cache otherwise."""
    global _pipeline_cache
    if _IS_STREAMLIT:
        @st.cache_resource
        def _cached():
            return _initialize_rag_pipeline_impl()
        return _cached()
    else:
        if _pipeline_cache is None:
            _pipeline_cache = _initialize_rag_pipeline_impl()
        return _pipeline_cache


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT MAIN APP
# ═══════════════════════════════════════════════════════════════════
if _IS_STREAMLIT:
    try:
        qa_chain = initialize_rag_pipeline()
    except Exception as e:
        st.error(f"خطأ فادح في تحميل النظام: {e}")
        st.stop()

    # ── Session state ────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Display chat history ─────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(convert_to_eastern_arabic(msg["content"]))

    # ── Handle new input ─────────────────────────────────────────
    if prompt_input := st.chat_input("اكتب سؤالك القانوني هنا…"):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("جاري التحليل القانوني…"):
                try:
                    # Build chat history from previous messages (exclude current question)
                    history = format_chat_history(st.session_state.messages[:-1])

                    # Invoke chain
                    result = qa_chain.invoke(
                        {"input": prompt_input, "chat_history": history}
                    )

                    if isinstance(result, dict):
                        response_text = result.get("answer", "")
                        source_docs = result.get("context", [])
                    else:
                        response_text = str(result)
                        source_docs = []

                    # Display answer
                    st.markdown(convert_to_eastern_arabic(response_text))

                    # Display sources
                    if source_docs:
                        seen: Set[str] = set()
                        unique_docs: List[Document] = []
                        for doc in source_docs:
                            art_num = str(doc.metadata.get("article_number", "")).strip()
                            if art_num and art_num not in seen:
                                seen.add(art_num)
                                unique_docs.append(doc)

                        st.markdown("---")
                        if unique_docs:
                            with st.expander(f"📚 المصادر المستخدمة ({len(unique_docs)} مادة)"):
                                st.markdown("### المواد القانونية المستخدمة في التحليل:")
                                st.markdown("---")
                                for doc in unique_docs:
                                    art_num = str(doc.metadata.get("article_number", "")).strip()
                                    law = doc.metadata.get("law_name", "")
                                    nature = doc.metadata.get("legal_nature", "")
                                    header = f"**المادة رقم {convert_to_eastern_arabic(art_num)}**"
                                    if law:
                                        header += f" — {law}"
                                    st.markdown(header)
                                    if nature:
                                        st.markdown(f"*الطبيعة القانونية: {nature}*")
                                    for line in doc.page_content.strip().split("\n"):
                                        line = line.strip()
                                        if line:
                                            st.markdown(convert_to_eastern_arabic(line))
                                    st.markdown("---")
                        else:
                            st.info("📌 لم يتم العثور على مصادر")
                    else:
                        st.info("📌 لم يتم العثور على مصادر")

                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"حدث خطأ: {e}")
