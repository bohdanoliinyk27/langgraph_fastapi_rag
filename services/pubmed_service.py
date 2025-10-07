import os
import requests
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET


class PubmedSearchService:
    """
    Мини-сервис для поиска по PubMed через NCBI eUtils (esearch + efetch).
    Возвращает список словарей со структурой:
    {
        "pmid": str, "title": str, "journal": str, "pdat": str,
        "abstract": str, "authors": List[str], "pmc": Optional[str],
        "full_text": str, "url": str
    }
    """

    def __init__(self, api_key: Optional[str] = None, db: str = "pubmed"):
        self.api_key = api_key or os.getenv("PUBMED_API_KEY", "")
        self.db = db

    def search(
        self,
        term: str,
        max_results: int = 10,
        timeout: int = 20
    ) -> List[Dict[str, Any]]:
        term_q = term.strip().replace(" ", "+")
        esearch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db={self.db}&retmax={max_results}&sort=relevance&retmode=json&term={term_q}"
        )
        print(esearch)
        if self.api_key:
            esearch += f"&api_key={self.api_key}"

        try:
            r = requests.get(esearch, timeout=timeout)
            print(r)
            r.raise_for_status()
            ids = (r.json().get("esearchresult", {}) or {}).get("idlist", []) or []
            if not ids:
                return []
            return self._fetch_details(ids, timeout=timeout)
        except Exception:
            return []

    # ---------------- internal helpers ----------------

    def _fetch_details(self, id_list: List[str], timeout: int = 25) -> List[Dict[str, Any]]:
        ids = ",".join(id_list)
        efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": self.db,
            "id": ids,
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            r = requests.get(efetch, params=params, timeout=timeout)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        for p in root.findall(".//PubmedArticle"):
            try:
                pmid = (p.findtext(".//PMID") or "").strip()
                title = "".join((p.find(".//ArticleTitle") or ET.Element("x")).itertext()).strip()
                journal = (p.findtext(".//Journal/Title") or "").strip()
                pdate = self._parse_date(p.find(".//ArticleDate")) or self._parse_date(p.find(".//PubDate")) or ""
                abstract = self._extract_abstract(p)
                authors = self._authors(p)

                # Try to fetch PMC full text (если доступен)
                pmc = None
                for aid in p.findall(".//ArticleIdList/ArticleId"):
                    if (aid.get("IdType") or "").lower() == "pmc":
                        pmc = (aid.text or "").strip()
                        break

                full_text = ""
                if pmc:
                    pmc_id = pmc.replace("PMC", "")
                    ft = self._fetch_pmc_fulltext(pmc_id, timeout=timeout)
                    # режем до 200k символов по твоему правилу
                    full_text = (ft or "")[:200_000] or "The full text is not available in the PMC library."

                out.append({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "pdat": pdate,
                    "abstract": abstract,
                    "authors": authors,
                    "pmc": pmc,
                    "full_text": full_text or "",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                })
            except Exception:
                continue

        return out

    def _extract_abstract(self, article_el: ET.Element) -> str:
        texts: List[str] = []
        for ab in article_el.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            txt = "".join(ab.itertext()).strip()
            texts.append(f"{label}: {txt}" if label else txt)
        s = "\n".join(texts)
        # нормализация спецсимволов
        return (
            s.replace("\u2009", " ")
             .replace("\u202f", " ")
             .replace("\u00a0", " ")
             .replace("\u2217", "*")
        )

    def _authors(self, article_el: ET.Element) -> List[str]:
        res: List[str] = []
        for a in article_el.findall(".//AuthorList/Author"):
            last = (a.findtext("LastName") or "").strip()
            fore = (a.findtext("ForeName") or "").strip()
            full = f"{fore} {last}".strip() if (fore or last) else (a.findtext("CollectiveName") or "").strip()
            if full:
                res.append(full)
        return res

    def _parse_date(self, date_el: Optional[ET.Element]) -> str:
        if date_el is None:
            return ""
        y = (date_el.findtext("Year") or "").strip()
        m = (date_el.findtext("Month") or "").strip()
        d = (date_el.findtext("Day") or "").strip()
        if y and m and d:
            m2 = m.zfill(2) if m.isdigit() else m
            d2 = d.zfill(2) if d.isdigit() else d
            return f"{y}-{m2}-{d2}"
        if y and m:
            m2 = m.zfill(2) if m.isdigit() else m
            return f"{y}-{m2}"
        return y

    def _fetch_pmc_fulltext(self, pmc_id: str, timeout: int = 25) -> str:
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}"
            if self.api_key:
                url += f"&api_key={self.api_key}"
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            body = root.find(".//body")
            if body is None:
                return "The publisher of this article does not allow access to the full text"

            chunks: List[str] = []
            for node in body.iter():
                if node.text:
                    chunks.append(node.text.strip())
                if node.tail:
                    chunks.append(node.tail.strip())
            return "\n".join([c for c in chunks if c])
        except Exception:
            return ""
