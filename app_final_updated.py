# -*- coding: utf-8 -*-
import os
import sys
import json
import re
from dotenv import load_dotenv
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Suppress progress bars from transformers/tqdm
os.environ['TRANSFORMERS_NO_PROGRESS_BAR'] = '1'
warnings.filterwarnings('ignore')

# 1. Loaders & Splitters
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Dict, Set
from rank_bm25 import BM25Okapi
import numpy as np

# 2. Vector Store & Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Reranker Imports
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 4. LLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ==========================================
# 📁 PATHS (use project-relative folders)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ==========================================
# 🔍 DETECT RUNTIME CONTEXT
# ==========================================
# True when run via `streamlit run`, False when imported by evaluate_rag.py etc.
_IS_STREAMLIT = False
try:
    import streamlit as st
    if hasattr(st, "runtime") and st.runtime.exists():
        _IS_STREAMLIT = True
except Exception:
    pass

# Only import streamlit features when running as a Streamlit app
if _IS_STREAMLIT:
    import streamlit as st

# ==========================================
# 🎨 UI SETUP (only when running as Streamlit app)
# ==========================================
if _IS_STREAMLIT:
    st.set_page_config(page_title="المساعد القانوني", page_icon="⚖️")

    # This CSS block fixes the "001" number issue and right alignment
    st.markdown("""
    <style>
        /* Force the main app container to be Right-to-Left */
        .stApp {
            direction: rtl;
            text-align: right;
        }
        
        /* Fix input fields to type from right */
        .stTextInput input {
            direction: rtl;
            text-align: right;
        }

        /* Fix chat messages alignment */
        .stChatMessage {
            direction: rtl;
            text-align: right;
        }
        
        /* Ensure proper paragraph spacing */
        .stMarkdown p {
            margin: 0.5em 0 !important;
            line-height: 1.6;
            word-spacing: 0.1em;
        }
        
        /* Ensure numbers display correctly in RTL */
        p, div, span, label {
            unicode-bidi: embed;
            direction: inherit;
            white-space: normal;
            word-wrap: break-word;
        }
        
        /* Force all content to respect RTL */
        * {
            direction: rtl !important;
        }
        
        /* Preserve line breaks and spacing */
        .stMarkdown pre {
            direction: rtl;
            text-align: right;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        /* Hide the "Deploy" button and standard menu for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
    """, unsafe_allow_html=True)

    st.title("⚖️ المساعد القانوني الذكي (دستور مصر)")

# Helper function (used by both Streamlit UI and evaluation)
def convert_to_eastern_arabic(text):
    """Converts 0123456789 to ٠١٢٣٤٥٦٧٨٩"""
    if not isinstance(text, str):
        return text 
    western_numerals = '0123456789'
    eastern_numerals = '٠١٢٣٤٥٦٧٨٩'
    translation_table = str.maketrans(western_numerals, eastern_numerals)
    return text.translate(translation_table)

# ==========================================
# 🚀 CACHED RESOURCE LOADING
# ==========================================
# Use @st.cache_resource only when running inside Streamlit;
# otherwise use a simple module-level cache so evaluate_rag.py can import this.
_pipeline_cache = None

def _initialize_rag_pipeline_impl():
    """Core pipeline builder — no Streamlit dependency."""
    print("🔄 Initializing system...")
    print("📥 Loading data...")

    # --- Arabic tokenizer utility (used by BM25 and metadata index) ---
    _ARABIC_STOPWORDS = {
        'في', 'من', 'على', 'إلى', 'عن', 'أن', 'هذا', 'هذه', 'التي', 'الذي',
        'ما', 'لا', 'أو', 'و', 'كل', 'ذلك', 'بين', 'كان', 'قد', 'هو', 'هي',
        'لم', 'بل', 'ثم', 'إذا', 'حتى', 'لكن', 'منه', 'فيه', 'عند', 'له',
        'بها', 'لها', 'منها', 'فيها', 'التى', 'الذى', 'ولا', 'وفى', 'كما',
        'تلك', 'هنا', 'أي', 'دون', 'ليس', 'إلا', 'أما', 'مع', 'عليه',
    }

    def arabic_tokenize(text: str) -> List[str]:
        """Tokenize Arabic text: remove diacritics, non-Arabic chars, and stopwords."""
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)          # strip tashkeel
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)            # keep Arabic only
        tokens = text.split()
        return [t for t in tokens if t not in _ARABIC_STOPWORDS and len(t) > 1]

    # 1. Load JSONs from ./data — propagate law_name to every article
    def load_json_folder(folder_path: str):
        all_items = []
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".json"):
                continue
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            wrapper_law_name = ""  # law_name from the wrapper object

            if isinstance(obj, list):
                # Could be [wrapper_dict_with_data] or [flat_articles]
                articles = []
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
                        # Flat article — guess law_name from article_id prefix
                        if not entry.get("_law_name"):
                            aid = entry.get("article_id", "")
                            if "CONST" in aid.upper():
                                entry["_law_name"] = "الدستور المصري"
                            else:
                                entry["_law_name"] = ""
                        articles.append(entry)
                all_items.extend(articles)
            elif isinstance(obj, dict):
                wrapper_law_name = obj.get("law_name", "")
                if "data" in obj and isinstance(obj["data"], list):
                    for art in obj["data"]:
                        art.setdefault("_law_name", wrapper_law_name)
                    all_items.extend(obj["data"])
                elif "articles" in obj and isinstance(obj["articles"], list):
                    for art in obj["articles"]:
                        art.setdefault("_law_name", wrapper_law_name)
                    all_items.extend(obj["articles"])
                else:
                    obj.setdefault("_law_name", wrapper_law_name)
                    all_items.append(obj)
            else:
                logger.warning(f"Unsupported JSON format in: {file_path}")
        return all_items

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    data = load_json_folder(DATA_DIR)

    # De-duplicate (article_id preferred, fallback to article_number)
    unique = {}
    for item in data:
        key = str(item.get("article_id") or item.get("article_number") or hash(json.dumps(item, ensure_ascii=False)))
        unique[key] = item
    data = list(unique.values())

    docs = []
    for item in data:
        article_number = item.get("article_number")
        original_text = item.get("original_text")
        simplified_summary = item.get("simplified_summary")

        if not article_number or not original_text or not simplified_summary:
            logger.warning("Skipping item with missing fields (article_number/original_text/simplified_summary)")
            continue

        # Resolve law_name: per-article "law_name" > propagated "_law_name"
        law_name = item.get("law_name") or item.get("_law_name", "")
        part_bab = item.get("part (Bab)", "")
        chapter_fasl = item.get("chapter (Fasl)", "")
        section = item.get("section", "")

        # Construct content — includes law_name for embedding quality
        page_content = f"""القانون: {law_name}
رقم المادة: {article_number}
الباب: {part_bab}
الفصل: {chapter_fasl}
النص الأصلي: {original_text}
الشرح المبسط: {simplified_summary}"""

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

    # 2. Embeddings
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1"
    )
    print("✅ Embeddings model ready")

    # 3. No splitting — keep articles as complete units
    chunks = docs

    # 4. Vector Store — persist once, reuse on subsequent runs (embeddings are saved)
    db_exists = os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)
    if db_exists:
        print("📦 Loading saved vector database (no re-embedding)...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        # Verify the DB has roughly the same number of docs
        stored_count = vectorstore._collection.count()
        if stored_count == 0 or abs(stored_count - len(chunks)) > 5:
            print(f"⚠️  DB doc count ({stored_count}) differs from data ({len(chunks)}). Rebuilding...")
            import shutil
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            db_exists = False
        else:
            print(f"✅ Loaded Chroma DB ({stored_count} vectors) — embeddings reused from disk")

    if not db_exists:
        print("🧱 Building vector database for the first time (creating embeddings)...")
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_DIR,
        )
        print(f"✅ Built & persisted Chroma DB ({len(chunks)} vectors)")

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    # 5. Create BM25 Keyword Retriever (with proper Arabic tokenization)
    class BM25Retriever(BaseRetriever):
        """BM25-based keyword retriever with Arabic-aware tokenization"""
        corpus_docs: List[Document]
        bm25: BM25Okapi = None
        tokenized_corpus: list = None
        k: int = 10

        class Config:
            arbitrary_types_allowed = True

        def __init__(self, **data):
            super().__init__(**data)
            self.tokenized_corpus = [arabic_tokenize(doc.page_content) for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            tokenized_query = arabic_tokenize(query)
            if not tokenized_query:
                return []
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:self.k]
            return [self.corpus_docs[i] for i in top_indices if scores[i] > 0]

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    bm25_retriever = BM25Retriever(corpus_docs=docs, k=10)
    print("✅ BM25 keyword retriever ready (Arabic tokenizer)")

    # 6. Create Metadata Filter Retriever (pre-built inverted index for speed)
    class MetadataFilterRetriever(BaseRetriever):
        """Fast metadata retriever using a pre-built inverted index."""
        corpus_docs: List[Document]
        keyword_index: Dict[str, Set[int]] = None   # token → set of doc indices
        law_name_index: Dict[str, Set[int]] = None
        k: int = 10

        class Config:
            arbitrary_types_allowed = True

        def __init__(self, **data):
            super().__init__(**data)
            # Build inverted indices at init time (once)
            self.keyword_index = defaultdict(set)
            self.law_name_index = defaultdict(set)
            for idx, doc in enumerate(self.corpus_docs):
                # Index keyword tokens
                kw_text = doc.metadata.get('keywords', '') + ' ' + doc.metadata.get('legal_nature', '')
                kw_text += ' ' + doc.metadata.get('part', '') + ' ' + doc.metadata.get('chapter', '')
                for token in arabic_tokenize(kw_text):
                    self.keyword_index[token].add(idx)
                # Index law_name tokens
                for token in arabic_tokenize(doc.metadata.get('law_name', '')):
                    self.law_name_index[token].add(idx)

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            query_tokens = arabic_tokenize(query)
            if not query_tokens:
                return []

            scores = defaultdict(float)
            for token in query_tokens:
                # Keyword / legal_nature / part / chapter match  (high weight)
                for idx in self.keyword_index.get(token, set()):
                    scores[idx] += 3.0
                # Law-name match (helps route to correct law)
                for idx in self.law_name_index.get(token, set()):
                    scores[idx] += 4.0

            if not scores:
                return []

            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.k]
            return [self.corpus_docs[idx] for idx, _ in top]

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    metadata_retriever = MetadataFilterRetriever(corpus_docs=docs, k=10)
    print("✅ Metadata filter retriever ready (inverted index)")

    # 7. Create Hybrid RRF Retriever (parallel execution)
    class HybridRRFRetriever(BaseRetriever):
        """Combines semantic, BM25, and metadata retrievers using Reciprocal Rank Fusion.
        All three sub-retrievers run **in parallel** via ThreadPoolExecutor."""
        semantic_retriever: BaseRetriever
        bm25_retriever: BM25Retriever
        metadata_retriever: MetadataFilterRetriever
        beta_semantic: float = 0.6
        beta_keyword: float = 0.2
        beta_metadata: float = 0.2
        k: int = 60   # RRF constant
        top_k: int = 12

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            # --- Run all three retrievers in parallel ---
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_semantic = pool.submit(self.semantic_retriever.invoke, query)
                fut_bm25     = pool.submit(self.bm25_retriever.invoke, query)
                fut_meta     = pool.submit(self.metadata_retriever.invoke, query)

                semantic_docs = fut_semantic.result()
                bm25_docs     = fut_bm25.result()
                metadata_docs = fut_meta.result()

            # --- Reciprocal Rank Fusion ---
            rrf_scores: Dict[str, float] = {}
            all_docs: Dict[str, Document] = {}

            for weight, doc_list in [
                (self.beta_semantic, semantic_docs),
                (self.beta_keyword,  bm25_docs),
                (self.beta_metadata, metadata_docs),
            ]:
                for rank, doc in enumerate(doc_list, start=1):
                    doc_id = doc.metadata.get('article_id') or doc.metadata.get('article_number') or str(hash(doc.page_content))
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight / (self.k + rank)
                    if doc_id not in all_docs:
                        all_docs[doc_id] = doc

            sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return [all_docs[did] for did, _ in sorted_ids[:self.top_k] if did in all_docs]

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            return self._get_relevant_documents(query, run_manager=run_manager)

    hybrid_retriever = HybridRRFRetriever(
        semantic_retriever=base_retriever,
        bm25_retriever=bm25_retriever,
        metadata_retriever=metadata_retriever,
        beta_semantic=0.5,
        beta_keyword=0.3,
        beta_metadata=0.2,
        k=60,
        top_k=12,
    )
    print("✅ Hybrid RRF retriever ready (parallel, β: sem=0.5, bm25=0.3, meta=0.2)")



    # 9. Reranker
    print("Loading reranker model...")
    local_model_path = r"D:\FOE\Senior 2\Graduation Project\Chatbot_me\reranker"
    
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Reranker path not found: {local_model_path}")

    model = HuggingFaceCrossEncoder(model_name=local_model_path)
    compressor = CrossEncoderReranker(model=model, top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )
    print("✅ Reranker model ready")

    # 7. LLM Configuration — tuned for legal accuracy with practical flexibility
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",  # Best Arabic comprehension on Groq
        temperature=0.2,       # Low for legal precision; >0 allows natural phrasing
        max_tokens=2048,       # Ensure complete legal answers are never truncated
        model_kwargs={
            "top_p": 0.85,     # Focused sampling — avoids hallucinated legal claims
        }
    )

# ==================================================
    # 🛠️ THE FIX: SEPARATE SYSTEM INSTRUCTIONS FROM USER INPUT
    # ==================================================
    
# ==================================================
    # 🧠 PROMPT ENGINEERING: DECISION TREE LOGIC
    # ==================================================
    
    system_instructions = """
<role>
أنت "المساعد القانوني الذكي"، مستشار قانوني متخصص في القوانين المصرية التالية:
- الدستور المصري
- القانون المدني المصري
- قانون العمل المصري
- قانون الأحوال الشخصية المصري
- قانون مكافحة جرائم تقنية المعلومات
- قانون الإجراءات الجنائية المصري

مهمتك الأساسية: الإجابة بدقة استناداً إلى "السياق التشريعي" المرفق أدناه.
عند وجود نص قانوني في السياق، هو مصدرك الأول والأهم.
</role>

<decision_logic>
حلّل سؤال المستخدم ثم اتبع أول حالة ينطبق شرطها:

━━━ الحالة ١ — الإجابة موجودة في السياق (الأولوية القصوى) ━━━
الشرط: توجد مادة أو أكثر في السياق تتناول موضوع السؤال بشكل مباشر أو وثيق الصلة.
الفعل:
• أجب من السياق مباشرةً دون مقدمات.
• وثّق كل معلومة بذكر اسم القانون ورقم المادة (مثال: «وفقاً للمادة (٥٢) من قانون العمل...»).
• استخرج ما يجيب السؤال تحديداً — لا تنسخ المادة كاملة.
• لا تضف معلومات من خارج السياق في هذه الحالة.

━━━ الحالة ٢ — السياق يغطي الموضوع جزئياً ━━━
الشرط: توجد مواد ذات صلة لكنها لا تجيب السؤال بالكامل.
الفعل:
• اذكر أولاً ما تنص عليه المواد المتاحة (مع التوثيق).
• ثم أضف توضيحاً عملياً مختصراً يساعد المستخدم، مع التنبيه بعبارة:
  «ملاحظة عملية:» أو «من الناحية التطبيقية:» قبل أي إضافة.
• لا تخترع أرقام مواد أو تنسب نصوصاً لقوانين لم ترد في السياق.

━━━ الحالة ٣ — السياق لا يحتوي الإجابة + السؤال إجرائي/عملي ━━━
الشرط: لا توجد مادة في السياق تتعلق بالموضوع، لكن السؤال عن إجراءات عملية (بلاغ، محضر، حادث، طلاق، تعامل مع الشرطة...).
الفعل:
• ابدأ بعبارة: «بناءً على الإجراءات القانونية المتعارف عليها في مصر (وليس استناداً لنص قانوني محدد من قاعدة البيانات):»
• قدم خطوات عملية مرقمة ومختصرة.
• لا تذكر أرقام مواد — لا تخترع مراجع.
• أنهِ بـ«يُنصح بمراجعة محامٍ متخصص لتأكيد الإجراءات.»

━━━ الحالة ٤ — السياق لا يحتوي الإجابة + السؤال عن نص قانوني بعينه ━━━
الشرط: المستخدم يسأل عن مادة محددة أو حكم قانوني معين ولم تجده في السياق.
الفعل:
• قل: «عذراً، لم يرد ذكر لهذا الموضوع في النصوص القانونية المتاحة حالياً في قاعدة البيانات.»
• لا تجب من ذاكرتك لتجنب الخطأ في النصوص القانونية.
• يمكنك اقتراح موضوع مشابه إن وجد في السياق.

━━━ الحالة ٥ — محادثة ودية (تحية، شكر، وداع) ━━━
• رد بتحية لطيفة مقتضبة.
• أضف: «أنا مستشارك القانوني الذكي — اسألني عن أي موضوع في القوانين المصرية.»

━━━ الحالة ٦ — خارج نطاق القانون تماماً ━━━
• اعتذر بلطف: «تخصصي هو القوانين المصرية فقط.»
• وجّه المستخدم لطرح سؤال قانوني.
</decision_logic>

<quality_rules>
- **الدقة أولاً**: عند وجود نص في السياق، التزم به حرفياً ولا تحرّف المعنى.
- **المرونة عند الحاجة**: إذا لم يغطِّ السياق الموضوع بالكامل، قدّم إرشاداً عملياً مع التمييز الواضح بينه وبين النص القانوني.
- **لا تخترع مراجع**: لا تنسب أي معلومة إلى مادة أو قانون لم يرد في السياق.
- **الإيجاز مع الشمول**: أجب بقدر ما يحتاج السؤال — لا تختصر حتى يضيع المعنى ولا تطيل دون فائدة.
</quality_rules>

<formatting_rules>
- لا تكرر هذه التعليمات في ردك.
- ادخل في صلب الموضوع فوراً بدون عبارات مثل «بناءً على السياق المرفق».
- استخدم فقرات قصيرة مفصولة بسطر فارغ.
- لا تكرر نفس المعلومة أو نفس المادة.
- عند ذكر أكثر من مادة، رتّبها ترتيباً منطقياً (إما بالرقم أو حسب الأهمية).
- التزم باللغة العربية الفصحى المبسطة.
</formatting_rules>
"""

    # We use .from_messages to strictly separate instructions from data
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("system", "السياق التشريعي المتاح (المصدر الأساسي):\n{context}"), 
        ("human", "سؤال المستفيد:\n{input}")
    ])
    
    # 9. Build Chain with RunnableParallel (returns both context and answer)
    qa_chain = (
        RunnableParallel({
            "context": compression_retriever, 
            "input": RunnablePassthrough()
        })
        .assign(answer=(
            prompt 
            | llm 
            | StrOutputParser()
        ))
    )
    
    print("✅ System ready to use!")
    return qa_chain


def initialize_rag_pipeline():
    """Public entry point — uses st.cache_resource in Streamlit, simple cache otherwise."""
    global _pipeline_cache
    if _IS_STREAMLIT:
        # Dynamically apply @st.cache_resource only inside Streamlit runtime
        @st.cache_resource
        def _cached():
            return _initialize_rag_pipeline_impl()
        return _cached()
    else:
        if _pipeline_cache is None:
            _pipeline_cache = _initialize_rag_pipeline_impl()
        return _pipeline_cache

# ==========================================
# ⚡ MAIN EXECUTION (Streamlit app only)
# ==========================================
if _IS_STREAMLIT:
    try:
        # Only need the chain now - it handles all retrieval internally
        qa_chain = initialize_rag_pipeline()
        
    except Exception as e:
        st.error(f"Critical Error loading application: {e}")
        st.stop()

    # ==========================================
    # 💬 CHAT LOOP
    # ==========================================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History (with Eastern Arabic numerals)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Convert to Eastern Arabic when displaying from history
            st.markdown(convert_to_eastern_arabic(message["content"]))

    # Handle New User Input
    if prompt_input := st.chat_input("اكتب سؤالك القانوني هنا..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("جاري التحليل القانوني..."):
                try:
                    # Invoke chain ONCE - returns Dict with 'context', 'input', and 'answer'
                    result = qa_chain.invoke(prompt_input)
                    
                    # Extract answer and context from result
                    response_text = result["answer"]
                    source_docs = result["context"]  # Context is already in the result!

                    # Display Answer
                    response_text_arabic = convert_to_eastern_arabic(response_text)
                    st.markdown(response_text_arabic)
                    
                    # Display Sources
                    if source_docs and len(source_docs) > 0:
                        print(f"✅ Found {len(source_docs)} documents")
                        # Deduplicate documents by article_number
                        seen_articles = set()
                        unique_docs = []
                        
                        for doc in source_docs:
                            article_num = str(doc.metadata.get('article_number', '')).strip()
                            if article_num and article_num not in seen_articles:
                                seen_articles.add(article_num)
                                unique_docs.append(doc)
                        
                        st.markdown("---")  # Separator before sources
                        
                        if unique_docs:
                            with st.expander(f"📚 المصادر المستخدمة ({len(unique_docs)} مادة)"):
                                st.markdown("### المواد القانونية المستخدمة في التحليل:")
                                st.markdown("---")
                                
                                for idx, doc in enumerate(unique_docs, 1):
                                    article_num = str(doc.metadata.get('article_number', '')).strip()
                                    legal_nature = doc.metadata.get('legal_nature', '')
                                    law_name = doc.metadata.get('law_name', '')
                                    
                                    if article_num:
                                        header = f"**المادة رقم {convert_to_eastern_arabic(article_num)}**"
                                        if law_name:
                                            header += f" — {law_name}"
                                        st.markdown(header)
                                        if legal_nature:
                                            st.markdown(f"*الطبيعة القانونية: {legal_nature}*")
                                        
                                        # Display article content
                                        content_lines = doc.page_content.strip().split('\n')
                                        for line in content_lines:
                                            line = line.strip()
                                            if line:
                                                st.markdown(convert_to_eastern_arabic(line))
                                        
                                        st.markdown("---")
                        else:
                            st.info("📌 لم يتم العثور على مصادر")
                    else:
                        st.info("📌 لم يتم العثور على مصادر")
                    
                    # Persist the raw answer to avoid double conversion glitches on rerun
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"حدث خطأ: {e}")