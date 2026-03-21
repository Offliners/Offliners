import datetime as dt
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from phraseg import Phraseg
from wordcloud import WordCloud

mpl.rcParams["figure.dpi"] = 300

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
MAX_TOTAL_RESULTS = 500
BATCH_SIZE = 100
TOP_K_DEFAULT = 3
README_START_MARKER = "<!-- ARXIV_DIGEST_TOP3_START -->"
README_END_MARKER = "<!-- ARXIV_DIGEST_TOP3_END -->"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def build_search_query() -> str:
    categories = os.getenv("ARXIV_CATEGORIES", "cs.*,cs.AR")
    terms = [x.strip() for x in categories.split(",") if x.strip()]
    if not terms:
        terms = ["cs.*", "cs.AR"]
    return " OR ".join(f"cat:{term}" for term in terms)


def fetch_entries(search_query: str) -> list[dict]:
    entries: list[dict] = []
    for start in range(0, MAX_TOTAL_RESULTS, BATCH_SIZE):
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "sortBy": "lastUpdatedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": BATCH_SIZE,
            }
        )
        with urllib.request.urlopen(f"{ARXIV_API_URL}?{params}", timeout=30) as response:
            xml_bytes = response.read()
        root = ET.fromstring(xml_bytes)
        page_entries = root.findall("atom:entry", ARXIV_NS)
        if not page_entries:
            break
        for entry in page_entries:
            updated = entry.findtext("atom:updated", default="", namespaces=ARXIV_NS)
            title = entry.findtext("atom:title", default="", namespaces=ARXIV_NS)
            summary = entry.findtext("atom:summary", default="", namespaces=ARXIV_NS)
            article_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
            if not updated:
                continue
            entries.append(
                {
                    "updated": updated.strip(),
                    "title": " ".join(title.split()),
                    "summary": " ".join(summary.split()),
                    "id": article_id.strip(),
                }
            )
    return entries


def in_digest_window(updated_at: str, now_utc: dt.datetime, hours: int) -> bool:
    updated_dt = dt.datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )
    lower_bound = now_utc - dt.timedelta(hours=hours)
    return lower_bound <= updated_dt <= now_utc


def top_digest_entries(entries: list[dict], top_k: int) -> list[dict]:
    """Same pool as wordcloud: already sorted by lastUpdatedDate descending from API."""
    return entries[:top_k]


def update_readme_top3(root_dir: Path, top_entries: list[dict]) -> None:
    readme_path = root_dir / "README.md"
    original = readme_path.read_text(encoding="utf-8")
    if README_START_MARKER not in original or README_END_MARKER not in original:
        raise RuntimeError("README markers for arXiv digest top papers are missing.")

    if top_entries:
        lines = []
        for idx, item in enumerate(top_entries, start=1):
            title = item.get("title", "Untitled paper")
            url = item.get("id", "").replace("http://", "https://")
            lines.append(f"{idx}. [{title}]({url})")
        body = "\n".join(lines)
    else:
        body = "No papers in current digest window (Computer Science & Hardware Architecture)."

    block = f"{README_START_MARKER}\n{body}\n{README_END_MARKER}"
    head, tail = original.split(README_START_MARKER, maxsplit=1)
    _, rest = tail.split(README_END_MARKER, maxsplit=1)
    updated = f"{head}{block}{rest}"
    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated README top papers: {readme_path.relative_to(root_dir)}")


def main() -> None:
    digest_hours = _env_int("DIGEST_WINDOW_HOURS", 24)
    top_k = _env_int("ARXIV_TOP_K", TOP_K_DEFAULT)
    now_utc = dt.datetime.now(dt.timezone.utc)
    search_query = build_search_query()
    print(f"Fetching arXiv entries with query: {search_query}")

    raw_entries = fetch_entries(search_query)
    print(f"Fetched {len(raw_entries)} entries from arXiv API")

    filtered_entries = [
        x
        for x in raw_entries
        if in_digest_window(x["updated"], now_utc=now_utc, hours=digest_hours)
    ]
    print(f"Entries in last {digest_hours}h: {len(filtered_entries)}")

    digest_top = top_digest_entries(filtered_entries, top_k=top_k)
    print(f"Top {top_k} papers (same scope as wordcloud): {len(digest_top)}")

    text_blob = "\n".join(
        f"{x['title']}\n{x['summary']}" for x in filtered_entries if x["title"] or x["summary"]
    )

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    output_path = script_dir / "wordcloud.png"
    font_path = script_dir / "NotoSansCJKtc-Medium.otf"

    if text_blob.strip():
        phraseg = Phraseg(text_blob, idf_chunk=300)
        result = phraseg.extract(result_word_minlen=1, merge_overlap=True)
    else:
        result = {"No new arXiv updates": 1.0}

    wordcloud = WordCloud(
        font_path=str(font_path),
        width=1800,
        height=1000,
        margin=1,
        background_color="white",
        collocations=False,
    ).fit_words(result)

    plt.figure(figsize=(18, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    update_readme_top3(root_dir=root_dir, top_entries=digest_top)
    print(f"Saved digest wordcloud to: {output_path.relative_to(root_dir)}")


if __name__ == "__main__":
    main()
