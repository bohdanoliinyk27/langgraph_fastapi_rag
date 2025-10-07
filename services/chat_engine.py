import os
import uuid
import base64
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Tuple
from copy import deepcopy
from dotenv import load_dotenv

from services.pinecone_client import index
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

# PubMed сервис — отдельный файл
from services.pubmed_service import PubmedSearchService

load_dotenv()

IMAGE_DIR = Path(__file__).parent / "generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_ENDPOINT = "https://api.tavily.com/search"

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)


def llm_chat(query: str) -> str:
    """Simple LLM chat wrapper."""
    return llm([HumanMessage(content=query)]).content


def image_generation(data: Any) -> str:
    """
    Call OpenAI DALL·E to generate an image from `data` (str or dict with 'text').
    Saves locally and returns the file path.
    """
    if isinstance(data, dict):
        prompt_text = data.get('text', "")
    elif isinstance(data, str):
        prompt_text = data
    else:
        raise TypeError(f"{type(data)}; Expected str or dict")

    prompt = prompt_text.strip() + " do not crop the image"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="auto",
        quality="auto",
        n=1,
    )
    img_data = resp.data[0].b64_json
    img_bytes = base64.b64decode(img_data)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"image_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    filepath = IMAGE_DIR / filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return str(filepath.resolve())


def wrap_node(fn, name: str):
    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = fn(deepcopy(state))
        trace = deepcopy(state.get("trace", []))
        output: Dict[str, Any] = {}
        for key in (
                "needs_image", "needs_chart", "use_rag", "chunks",
                "kept_summaries", "answer", "question",
                "needs_web", "web_chunks",
                "needs_pubmed", "pubmed_query", "pubmed_chunks"
        ):
            if key in new_state and new_state.get(key) != state.get(key):
                output[key] = new_state[key]
        trace.append({"node": name, "output": output})
        new_state["trace"] = trace
        return new_state
    return wrapped


# ========= Chart generation nodes =========
def needs_chart(state: "GraphState") -> "GraphState":
    prompt = (
        f"Decide if the user wants a chart not image!. Answer 'yes' or 'no'.\n"
        f"User request: {state['question']}"
    )
    resp = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip().lower()
    return {**state, "needs_chart": resp.startswith("yes")}


def generate_chart(state: "GraphState") -> "GraphState":
    instructions = "Build GET URL to https://quickchart.io/chart with Chart.js config; return only the URL."
    prompt = instructions + f"\nUser request: {state['question']}"
    url = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()
    return {**state, "answer": url}


def needs_image(state: "GraphState") -> "GraphState":
    prompt = (
        f"Decide if the user wants an image not chart. Answer 'yes' or 'no'.\n"
        f"User request: {state['question']}"
    )
    resp = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip().lower()
    return {**state, "needs_image": resp.startswith("yes")}


def generate_image(state: "GraphState") -> "GraphState":
    link = image_generation(state['question'])
    return {**state, "answer": link}


class GraphState(TypedDict, total=False):
    question: str
    attached: bool
    namespaces: List[str]
    history: List[Tuple[str, str]]
    needs_chart: bool
    needs_image: bool
    use_rag: bool
    chunks: List[str]
    kept_summaries: List[str]
    # --- Web branch ---
    force_web: bool
    needs_web: bool
    web_docs: List[Dict[str, str]]  # [{"url","title","content"}]
    web_chunks: List[str]
    # --- PubMed branch ---
    force_pubmed: bool
    needs_pubmed: bool
    pubmed_query: str
    pubmed_docs: List[Dict[str, Any]]
    pubmed_chunks: List[str]
    # --- Common ---
    answer: str
    trace: List[Dict[str, Any]]
    attempts: int


def check_rag_usage(state: GraphState) -> GraphState:
    summary_dir = Path("summaries")
    summaries: List[str] = []
    for ns in state.get("namespaces", []):
        summary_file = summary_dir / f"{Path(ns).stem}.txt"
        if summary_file.exists():
            content = summary_file.read_text(encoding="utf-8")
            summaries.append(f"{Path(ns).name}: {content}")
    prompt = (
        f"User query: {state['question']}\n"
        f"Summaries: {'; '.join(summaries)}\n?"
        "Is this question related to this file (or files)? "
        "Can it be answered using only the information in the file(s)? "
        "If user mentioned file in the query always answer yes"
        "If the file summaries do not contain relevant information to the user’s query, answer \"no.\" yes/no"
    )
    resp = ChatOpenAI(model_name="gpt-4o", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip().lower()
    return {**state, "use_rag": resp.startswith("yes"), "chunks": [], "kept_summaries": summaries}


def retrieve(state: GraphState) -> GraphState:
    all_chunks: List[str] = []
    for ns in state.get("namespaces", []):
        resp = index.search(
            namespace=ns,
            query={"inputs": {"text": state['question']}, "top_k": 5},
            fields=["chunk_text"]
        )
        hits = resp.get("result", {}).get("hits", [])
        all_chunks.extend(h["fields"]["chunk_text"] for h in hits)
    return {**state, "chunks": all_chunks}


def grade_documents(state: GraphState) -> GraphState:
    kept: List[str] = []
    grader = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
    for chunk in state.get("chunks", []):
        prompt = (
            f"Text: {chunk}\n"
            f"Question: {state['question']}\n"
            "Help answer? yes/no."
        )
        reply = grader([HumanMessage(content=prompt)]).content.strip().lower()
        if reply.startswith("yes"):
            kept.append(chunk)

    summary_dir = Path("summaries")
    summaries: List[str] = []
    for ns in state.get("namespaces", []):
        summary_file = summary_dir / f"{Path(ns).stem}.txt"
        if summary_file.exists():
            content = summary_file.read_text(encoding="utf-8")
            summaries.append(f"{Path(ns).name}: {content}")

    filtered: List[str] = []
    for summ in summaries:
        prompt = (
            f"Summary: {summ}\n"
            f"Question: {state['question']}\n"
            "Is this question related to this Summary? "
            "Can it be answered using only the information in the Summary? "
            "If user mentioned file in the query always answer yes "
            "If the file summaries do not contain relevant information to the user’s query, answer \"no.\" yes/no"
        )
        reply = grader([HumanMessage(content=prompt)]).content.strip().lower()
        if reply.startswith("yes"):
            filtered.append(summ)

    return {**state, "chunks": kept, "kept_summaries": filtered}


def transform_query(state: GraphState) -> GraphState:
    prompt = f"Rewrite query given summaries {'; '.join(state.get('kept_summaries', []))}: {state['question']}"
    new_q = ChatOpenAI(model_name="gpt-4o", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()
    return {**state, "question": new_q, "attempts": state.get("attempts", 0) + 1}


def generate_final(state: GraphState) -> GraphState:
    convo = "\n".join(f"{r.title()}: {m}" for r, m in state.get("history", []))
    context_items = (
        state.get('chunks', [])
        + state.get('kept_summaries', [])
        + state.get('web_chunks', [])
        + state.get('pubmed_chunks', [])
    )
    context = "\n".join(context_items)
    prompt = f"History: {convo}\nContext:\n{context}\n\nQuestion: {state['question']}\nAnswer:"
    return {**state, "answer": llm_chat(prompt)}


def llm_generate(state: GraphState) -> GraphState:
    convo = "\n".join(f"{role.title()}: {msg}" for role, msg in state.get("history", []))
    prompt = f"History:\n{convo}\n\nQuestion: {state['question']}\nAnswer:"
    return {**state, "answer": llm_chat(prompt)}


def add_intext_citations(state: GraphState) -> GraphState:
    previous_answer = state.get("answer", "")
    summaries = state.get("kept_summaries", [])
    files = state.get("namespaces", [])

    prompt = (
        "You are an assistant whose task is to add in-text citations to an already generated answer. "
        "For each paragraph in the answer, insert a citation in square brackets indicating the source file. "
        "Decide which file each paragraph is drawn from by using the provided summaries.\n\n"
        "Currently attached files:\n" +
        "\n".join(f"- {f}" for f in files) +
        "\n\n"
        "File summaries:\n" +
        "\n".join(summaries) +
        "\n\n"
        "Answer to annotate:\n" +
        previous_answer +
        "\n\n"
        "Rewrite the answer, preserving content but adding [filename.ext] after each paragraph."
    )

    annotated = ChatOpenAI(model_name="gpt-4o", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()

    return {**state, "answer": annotated}


# ========= Web Search helpers & nodes =========
def web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Returns a list of dicts: [{"url": "...", "title": "...", "content": "..."}]
    Uses Tavily if TAVILY_API_KEY provided; otherwise returns [].
    """
    if not TAVILY_API_KEY:
        return []
    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_images": False,
            "max_results": max_results,
        }
        r = requests.post(TAVILY_ENDPOINT, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("content", "")  # summary/snippet
            })
        return results
    except Exception:
        return []


def needs_web(state: GraphState) -> GraphState:
    if state.get("force_web"):
        return {**state, "needs_web": True}

    prompt = (
        "Answer strictly yes/no.\n"
        "Should I use live web search to answer the user's request?\n\n"
        f"User: {state['question']}\n"
        "Say yes for: recent news, current prices/schedules, changing facts, niche details, citations/links.\n"
        "Otherwise no."
    )
    resp = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip().lower()
    return {**state, "needs_web": resp.startswith("yes")}


def web_search(state: GraphState) -> GraphState:
    prompt = (
        "Create 2-3 distinct web search queries for the given question. "
        "Return them as a JSON array of strings, nothing else.\n\n"
        f"Question: {state['question']}"
    )
    q_json = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()
    try:
        queries = json.loads(q_json)
        if not isinstance(queries, list):
            queries = [state["question"]]
    except Exception:
        queries = [state["question"]]

    gathered: List[Dict[str, str]] = []
    seen_urls = set()
    for q in queries[:3]:
        results = web_search_tool(q, max_results=5)
        for r in results:
            u = r.get("url", "")
            if u and u not in seen_urls:
                seen_urls.add(u)
                gathered.append(r)

    web_chunks: List[str] = []
    for d in gathered:
        title = d.get("title", "").strip()
        content = d.get("content", "").strip()
        url = d.get("url", "").strip()
        if content or title:
            web_chunks.append(f"{title}\n{content}\nSource: {url}")

    return {**state, "web_docs": gathered, "web_chunks": web_chunks}


def grade_web_results(state: GraphState) -> GraphState:
    kept: List[str] = []
    grader = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
    for chunk in state.get("web_chunks", []):
        prompt = (
            f"Text:\n{chunk}\n\n"
            f"Question: {state['question']}\n"
            "Does this help answer? yes/no (strict)"
        )
        reply = grader([HumanMessage(content=prompt)]).content.strip().lower()
        if reply.startswith("yes"):
            kept.append(chunk)
    return {**state, "web_chunks": kept}


def add_web_citations(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    sources = "\n".join(state.get("web_chunks", []))
    if not answer or not sources:
        return state

    prompt = (
        "Annotate the answer with in-text citations to the most relevant 'Source: URL' lines from the provided sources. "
        "Append [URL] at the end of each paragraph that relies on the web. Keep the text intact otherwise.\n\n"
        f"Sources:\n{sources}\n\nAnswer:\n{answer}"
    )
    annotated = ChatOpenAI(model_name="gpt-4o", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()
    return {**state, "answer": annotated}


# ========= PubMed nodes =========
def needs_pubmed(state: GraphState) -> GraphState:
    if state.get("force_pubmed"):
        return {**state, "needs_pubmed": True}

    q = state["question"].lower()
    heuristics = any(
        kw in q for kw in [
            "pubmed", "pmid", "pmc", "randomized", "double-blind", "systematic review",
            "meta-analysis", "clinical trial", "guideline", "rct", "medical", "oncology",
            "cardiology", "diabetes", "covid", "copd", "pregnan", "neonatal", "adverse event"
        ]
    )
    if heuristics:
        return {**state, "needs_pubmed": True}

    prompt = (
        "Answer strictly yes/no.\n"
        "Is the user's request medical or biomedical/clinical and likely answerable via PubMed literature search?\n"
        f"User: {state['question']}"
    )
    resp = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip().lower()
    return {**state, "needs_pubmed": resp.startswith("yes")}

def pubmed_make_query(state: GraphState) -> GraphState:
    user_q = state["question"]

    system_instructions = (
        "You are a PubMed query composer. Craft a precise Entrez query using field tags.\n"
        "Guidelines:\n"
        "1) Generate citations in the final answer as [1],[2] etc (information only for downstream),\n"
        "   but HERE you must output only the query string.\n"
        "2) Do not include any References section.\n"
        "3) Extract all info from the user's prompt: dates/periods/authors/journals/affiliations/mesh terms.\n"
        "4) Use fields: [Author], [Title], [Abstract], [Journal], [Affiliation], [PDAT], [Mesh Terms].\n"
        "5) Use operators AND/OR/NOT and add filter hasabstract.\n"
        "6) Put all dates into [PDAT] range: YYYY/MM/DD:YYYY/MM/DD[PDAT].\n"
        "7) Output ONLY the PubMed query on a single line. No backticks, no quotes, no commentary."
    )

    prompt = (
        f"{system_instructions}\n\n"
        f"User prompt:\n{user_q}\n\n"
        "Return only the PubMed query:"
    )

    try:
        query = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)(
            [HumanMessage(content=prompt)]
        ).content.strip()
        if query.startswith("```"):
            query = query.strip("` \n")
            parts = [line for line in query.splitlines() if line.strip() and not line.strip().startswith("```")]
            if parts:
                query = parts[-1].strip()
        if not query:
            query = user_q
    except Exception:
        query = user_q

    return {**state, "pubmed_query": query}



def pubmed_search(state: GraphState) -> GraphState:
    service = PubmedSearchService()
    articles = service.search(state["pubmed_query"], max_results=10)
    chunks: List[str] = []
    for a in articles:
        title = a.get("title", "").strip()
        abstract = a.get("abstract", "").strip()
        pdat = a.get("pdat", "")
        pmid = a.get("pmid", "")
        url = a.get("url", "")
        piece = f"{title}\n{abstract}\nPublished: {pdat}\nPMID: {pmid}\nSource: {url}".strip()
        if piece:
            chunks.append(piece)

    return {**state, "pubmed_docs": articles, "pubmed_chunks": chunks}


def grade_pubmed_results(state: GraphState) -> GraphState:
    kept: List[str] = []
    grader = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
    for chunk in state.get("pubmed_chunks", []):
        prompt = (
            f"Text:\n{chunk}\n\n"
            f"Question: {state['question']}\n"
            "Does this help answer? yes/no (strict)"
        )
        reply = grader([HumanMessage(content=prompt)]).content.strip().lower()
        if reply.startswith("yes"):
            kept.append(chunk)
    return {**state, "pubmed_chunks": kept}


def add_pubmed_citations(state: GraphState) -> GraphState:
    answer = state.get("answer", "")
    sources = "\n".join(state.get("pubmed_chunks", []))
    if not answer or not sources:
        return state

    prompt = (
        "Annotate the answer with in-text citations to most relevant 'PMID: N' lines from sources. "
        "Append [PMID: N] at the end of each paragraph that relies on PubMed. Keep the text intact otherwise.\n\n"
        f"Sources:\n{sources}\n\nAnswer:\n{answer}"
    )
    annotated = ChatOpenAI(model_name="gpt-4o", temperature=0.0)(
        [HumanMessage(content=prompt)]
    ).content.strip()
    return {**state, "answer": annotated}


# ========= Route helpers =========
def decide_after_grade(state: GraphState) -> str:
    if state.get("chunks"):
        return "generate_final"
    if state.get("kept_summaries"):
        return "transform_query"
    return "needs_web"


def decide_route(state: GraphState) -> str:
    if state.get("needs_image"):
        return "generate_image"
    if state.get("needs_chart"):
        return "generate_chart"
    if state.get("attached"):
        return "check_rag"
    # если нет файлов — смотрим PubMed
    return "needs_pubmed"


# ========= Build & compile graph =========
graph = StateGraph(GraphState)
for name, fn in [
    ("needs_image", needs_image),
    ("generate_image", generate_image),
    ("needs_chart", needs_chart),
    ("generate_chart", generate_chart),

    # PubMed
    ("needs_pubmed", needs_pubmed),
    ("pubmed_make_query", pubmed_make_query),
    ("pubmed_search", pubmed_search),
    ("grade_pubmed", grade_pubmed_results),

    # Web
    ("needs_web", needs_web),
    ("web_search", web_search),
    ("grade_web", grade_web_results),

    # RAG
    ("check_rag", check_rag_usage),
    ("retrieve", retrieve),
    ("grade_documents", grade_documents),
    ("transform_query", transform_query),

    # Gen/citations
    ("generate_final", generate_final),
    ("llm_generate", llm_generate),
    ("add_citations", add_intext_citations),
    ("add_pubmed_citations", add_pubmed_citations),
    ("add_web_citations", add_web_citations),
]:
    graph.add_node(name, wrap_node(fn, name))

# Start: image → chart → route
graph.add_edge(START, "needs_image")
graph.add_conditional_edges(
    "needs_image",
    lambda s: "generate_image" if s.get("needs_image") else "needs_chart",
    {"generate_image": "generate_image", "needs_chart": "needs_chart"}
)

graph.add_conditional_edges(
    "needs_chart",
    decide_route,
    {
        "generate_image": "generate_image",
        "generate_chart": "generate_chart",
        "check_rag": "check_rag",
        "needs_pubmed": "needs_pubmed",
    }
)

# RAG branch
graph.add_conditional_edges(
    "check_rag",
    lambda s: "retrieve" if s.get("use_rag") else "needs_pubmed",
    {"retrieve": "retrieve", "needs_pubmed": "needs_pubmed"}
)
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_after_grade,
    {"transform_query": "transform_query", "generate_final": "generate_final", "needs_web": "needs_web"}
)
graph.add_edge("transform_query", "retrieve")

# PubMed branch
graph.add_conditional_edges(
    "needs_pubmed",
    lambda s: "pubmed_make_query" if s.get("needs_pubmed") else "needs_web",
    {"pubmed_make_query": "pubmed_make_query", "needs_web": "needs_web"}
)
graph.add_edge("pubmed_make_query", "pubmed_search")  # <--- НОВОЕ
graph.add_edge("pubmed_search", "grade_pubmed")
graph.add_conditional_edges(
    "grade_pubmed",
    lambda s: "generate_final" if s.get("pubmed_chunks") else "needs_web",
    {"generate_final": "generate_final", "needs_web": "needs_web"}
)

# Web branch
graph.add_conditional_edges(
    "needs_web",
    lambda s: "web_search" if s.get("needs_web") else "llm_generate",
    {"web_search": "web_search", "llm_generate": "llm_generate"}
)
graph.add_edge("web_search", "grade_web")
graph.add_conditional_edges(
    "grade_web",
    lambda s: "generate_final" if s.get("web_chunks") else "llm_generate",
    {"generate_final": "generate_final", "llm_generate": "llm_generate"}
)

# Finishing touches (локальные → PubMed → Web)
graph.add_edge("generate_final", "add_citations")
graph.add_edge("add_citations", "add_pubmed_citations")
graph.add_edge("add_pubmed_citations", "add_web_citations")
graph.add_edge("add_web_citations", END)

# Terminal edges
graph.add_edge("generate_image", END)
graph.add_edge("generate_chart", END)

compiled_graph = graph.compile()


# ========= Public API =========
def ask_question(
    query: str,
    attached: List[str],
    history: Optional[List[Tuple[str, str]]] = None,
    force_web: bool = False,
    force_pubmed: bool = False
) -> Dict[str, Any]:
    state: GraphState = {
        "question": query,
        "attached": bool(attached),
        "namespaces": attached or [],
        "attempts": 0,
        "force_web": bool(force_web),
        "force_pubmed": bool(force_pubmed),
        "trace": [],
    }
    if history is not None:
        state["history"] = history
    out = compiled_graph.invoke(state)
    return {"answer": out.get("answer", ""), "trace": out.get("trace", [])}
