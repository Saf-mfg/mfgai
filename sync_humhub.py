from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pdfplumber
import io
import uuid
from urllib.parse import urljoin
from rag_db import collection

BASE_URL = "https://hub.mfgsolicitors.com"
SPACE_URL = "https://hub.mfgsolicitors.com/s/firm-policies/wiki/overview/list-categories"


# =========================
# PDF HELPERS
# =========================

def is_pdf(content: bytes):
    return content.startswith(b"%PDF")


def extract_text(pdf_bytes):
    text = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    return text.strip()


def chunk_text(text, chunk_size=1000):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


def extract_pdf_candidates(html):

    soup = BeautifulSoup(html, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):

        href = a["href"]

        if "file/file/download" in href:
            links.append(urljoin(BASE_URL, href))

        elif "file/download" in href:
            links.append(urljoin(BASE_URL, href))

        elif "file/view" in href:
            links.append(urljoin(BASE_URL, href))

        if a.get("data-file-url"):
            links.append(urljoin(BASE_URL, a["data-file-url"]))

    seen = set()
    unique = []

    for l in links:
        if l not in seen:
            unique.append(l)
            seen.add(l)

    return unique


def get_popup_download(page):

    selectors = [
        "a[data-file-download]",
        "a[data-file-url]"
    ]

    for selector in selectors:

        try:

            page.wait_for_selector(
                selector,
                timeout=1000
            )

            btn = page.locator(selector).first

            url = (
                btn.get_attribute("data-file-url")
                or btn.get_attribute("href")
            )

            if url:
                return urljoin(BASE_URL, url)

        except:
            pass

    return None


# =========================
# MAIN
# =========================

def run():

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=False)

        context = browser.new_context()

        page = context.new_page()

        # LOGIN
        page.goto(
            "https://hub.mfgsolicitors.com/user/auth/login"
        )

        input("🔐 Login manually then press ENTER...")

        print("🚀 Starting crawl...")

        # LOAD OVERVIEW
        page.goto(SPACE_URL)

        page.wait_for_load_state("networkidle")

        soup = BeautifulSoup(
            page.content(),
            "html.parser"
        )

        links = []

        for a in soup.find_all("a", href=True):

            href = a["href"]

            if not href.startswith(
                "/s/firm-policies/wiki/"
            ):
                continue

            if any(
                x in href
                for x in [
                    "edit",
                    "history",
                    "overview"
                ]
            ):
                continue

            full = urljoin(BASE_URL, href)

            if full not in links:
                links.append(full)

        print("Found wiki pages:", len(links))

        # =========================
        # PROCESS PAGES
        # =========================

        for url in links:

            print("\nPAGE:", url)

            try:

                page.goto(url)

                page.wait_for_load_state(
                    "domcontentloaded"
                )

                candidates = extract_pdf_candidates(
                    page.content()
                )

                if not candidates:
                    print("❌ No PDF candidates found")
                    continue

                stored_count = 0

                for pdf_url in candidates:

                    print("TRY:", pdf_url)

                    try:

                        # open viewer page
                        page.goto(pdf_url)

                        page.wait_for_load_state(
                            "domcontentloaded"
                        )

                        download_url = get_popup_download(
                            page
                        )

                        if not download_url:
                            print(
                                "❌ no download button found"
                            )
                            continue

                        print(
                            "🔁 CLICKED DOWNLOAD:",
                            download_url
                        )

                        response = context.request.get(
                            download_url
                        )

                        if response.status != 200:
                            print(
                                "❌ download failed:",
                                response.status
                            )
                            continue

                        pdf_bytes = response.body()

                        if not is_pdf(pdf_bytes):
                            print("❌ not a pdf")
                            continue

                        text = extract_text(pdf_bytes)

                        if not text:
                            print("⚠️ empty pdf")
                            continue

                        chunks = chunk_text(text)

                        collection.add(
                            documents=chunks,
                            ids=[str(uuid.uuid4()) for _ in chunks],
                                metadatas=[
                                    {
                                        "wiki_page": url,
                                        "pdf_url": download_url
                                    }
                                     for _ in chunks
                                ]
                        )

                        stored_count += len(chunks)

                        print(f"✅ stored{len(chunks)}chunks")

                    except Exception as e:
                        print(
                            "❌ candidate error:",
                            e
                        )

                print(
                    f"📦 Stored {stored_count} PDFs from page"
                )

            except Exception as e:
                print(
                    "❌ page error:",
                    e
                )

        browser.close()


if __name__ == "__main__":
    run()