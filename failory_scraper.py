"""
Failory.com Startup Scraper — SEA Edition
==========================================
Verified against live HTML of https://www.failory.com/startups/thailand

TOP 10 card structure (confirmed):
  <h3>1. <a href="https://company.com/?ref=failory">Name</a></h3>
  -- sibling text node: "Delivery Service E-Commerce Food Delivery Taxi Service"
  -- sibling <p> with <a href="/startups/e-commerce"> tags = industry links
  -- sibling <p> = description
  -- sibling <table>:
       Headquarters        | Bangkok, ...
       Year Founded        | 2010
       Founders            | Yod Chinsupakul
       Funding Amount      | $375M
       Startup Size        | Major Organization (1,001-5,000)   ← excluded per spec
       Last Funding Status | Series B
       Top Investors       | GIC, LINE Plus Corporation, ...

SUMMARY TABLE columns (confirmed):
  Startup (logo+name+link) | Industry | Year Founded | Amount Raised | Last Funding Round

Output columns:
  source_section, rank, country, name, url, industry,
  headquarters, year_founded, founders, funding_amount,
  last_funding_status, top_investors

Install:
    pip install playwright pandas openpyxl
    playwright install chromium

Run:
    python scrape_failory.py
"""

import asyncio
import csv
import os
import re
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

# ── Config ────────────────────────────────────────────────────────────────────
PAGES = {
    "singapore":   "https://www.failory.com/startups/singapore",
    "malaysia":    "https://www.failory.com/startups/malaysia",
    "indonesia":   "https://www.failory.com/startups/indonesia",
    "vietnam":     "https://www.failory.com/startups/vietnam",
    "thailand":    "https://www.failory.com/startups/thailand",
    "philippines": "https://www.failory.com/startups/philippines",
}

COMBINED_LABEL = "sea_all"
OUTPUT_DIR     = "SEA"

HEADLESS       = True
LOAD_WAIT      = 6_000  # ms

# ── Columns  (startup_size excluded per spec) ─────────────────────────────────
COLUMNS = [
    "source_section",
    "rank",
    "country",
    "name",
    "url",
    "industry",
    "headquarters",
    "year_founded",
    "founders",
    "funding_amount",
    "last_funding_status",
    "top_investors",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def strip_ref(url: str) -> str:
    return re.sub(r"[?&]ref=failory[^&]*", "", url).strip().rstrip("?&")

def empty_top10(rank, country) -> dict:
    return {c: "" for c in COLUMNS} | {"source_section": "top10",
                                        "rank": str(rank),
                                        "country": country}

def empty_table(rank, country) -> dict:
    return {c: "" for c in COLUMNS} | {"source_section": "table",
                                        "rank": str(rank),
                                        "country": country}


# ── TOP-10 extraction ─────────────────────────────────────────────────────────

async def extract_top10(page, country: str) -> list[dict]:
    """
    For each h3 matching "N. Company Name":
      - Extract name + URL from the <a> inside the h3
      - Walk nextElementSibling up to 15 steps:
          * <p> containing <a href="/startups/..."> → industry tags
          * <table> → parse every row as label:value
      - Map exact Failory table labels to output columns
    """
    records = []
    h3s = await page.query_selector_all("h3")

    for h3 in h3s:
        raw = clean(await h3.inner_text())
        m = re.match(r"^(\d+)\.\s+(.+)$", raw)
        if not m:
            continue

        rank = m.group(1)
        name = clean(m.group(2))

        # URL — <a> inside the h3
        url = await h3.evaluate(
            "el => { const a = el.querySelector('a'); return a ? a.href : ''; }"
        )
        url = strip_ref(url)

        # Walk siblings: collect industry <a> tags + the metadata <table>
        data = await page.evaluate("""(h3) => {
            const out = { industry: [], table: {} };
            let el = h3.nextElementSibling;
            let steps = 0;

            while (el && steps < 15) {
                const tag = el.tagName;

                // Stop at the next top-10 heading
                if (tag === "H3" && /^\\d+\\.\\s/.test(el.innerText.trim())) break;

                // <p> — collect /startups/ tag links as industry
                if (tag === "P") {
                    el.querySelectorAll('a[href*="/startups/"]').forEach(a => {
                        const t = a.innerText.trim();
                        // Exclude long navigation phrases
                        if (t && t.length < 50 && !t.includes("Startup")) {
                            out.industry.push(t);
                        }
                    });
                }

                // <table> — parse label:value rows
                if (tag === "TABLE") {
                    el.querySelectorAll("tr").forEach(row => {
                        const cells = row.querySelectorAll("td, th");
                        if (cells.length >= 2) {
                            // Keep original label text for exact matching below
                            const label = cells[0].innerText.trim();
                            const value = cells[1].innerText.trim();
                            out.table[label] = value;
                        }
                    });
                    // Only need the first table (the metadata one)
                    break;
                }

                el = el.nextElementSibling;
                steps++;
            }
            return out;
        }""", h3)

        t = data.get("table", {})

        # Deduplicate industry tags
        seen, tags = set(), []
        for tag in data.get("industry", []):
            k = tag.lower()
            if k not in seen:
                seen.add(k)
                tags.append(tag)

        record = empty_top10(rank, country)
        record.update({
            "name":               name,
            "url":                url,
            "industry":           ", ".join(tags),
            # Exact Failory table label strings (confirmed from live HTML):
            "headquarters":       clean(t.get("Headquarters", "")),
            "year_founded":       clean(t.get("Year Founded", "")),
            "founders":           clean(t.get("Founders", "")),
            "funding_amount":     clean(t.get("Funding Amount", "")),
            # "Startup Size" intentionally excluded
            "last_funding_status": clean(t.get("Last Funding Status", "")),
            "top_investors":      clean(t.get("Top Investors", "")),
        })
        records.append(record)

    return records


# ── SUMMARY TABLE extraction ───────────────────────────────────────────────────

async def extract_summary_table(page, country: str,
                                 top10_names_lower: set,
                                 rank_start: int) -> list[dict]:
    """
    Confirmed summary table header: Startup | Industry | Year Founded |
                                    Amount Raised | Last Funding Round

    Multiple <table> elements exist on the page (one per top-10 card + the
    summary table). We identify the summary table by its first <thead th>
    being "Startup" (case-insensitive). Fallback: largest table by row count.
    """
    records = []
    tables  = await page.query_selector_all("table")
    summary = None

    # Primary: find table whose first header cell is "Startup"
    for tbl in tables:
        first_th = await tbl.query_selector("thead th:first-child, thead td:first-child")
        if first_th:
            txt = clean(await first_th.inner_text()).lower()
            if txt == "startup":
                summary = tbl
                break

    # Fallback: largest table by tbody row count
    if not summary:
        best = 0
        for tbl in tables:
            rows = await tbl.query_selector_all("tbody tr")
            if len(rows) > best:
                best = len(rows)
                summary = tbl

    if not summary:
        print("  ✗ Summary table not found.")
        return records

    tbody_rows = await summary.query_selector_all("tbody tr")
    rank = rank_start

    for row in tbody_rows:
        cells = await row.query_selector_all("td")
        if len(cells) < 4:
            continue

        name = clean(await cells[0].inner_text())
        if not name:
            continue
        if name.lower() in top10_names_lower:
            continue

        url = await cells[0].evaluate(
            "el => { const a = el.querySelector('a'); return a ? a.href : ''; }"
        )
        url = strip_ref(url)

        # Confirmed column order: Startup | Industry | Year Founded |
        #                         Amount Raised | Last Funding Round
        industry     = clean(await cells[1].inner_text()) if len(cells) > 1 else ""
        year_founded = clean(await cells[2].inner_text()) if len(cells) > 2 else ""
        funding      = clean(await cells[3].inner_text()) if len(cells) > 3 else ""
        last_round   = clean(await cells[4].inner_text()) if len(cells) > 4 else ""

        record = empty_table(rank, country)
        record.update({
            "name":               name,
            "url":                url,
            "industry":           industry,
            "year_founded":       year_founded,
            "funding_amount":     funding,
            "last_funding_status": last_round,
            # headquarters, founders, top_investors not available in summary table
        })
        records.append(record)
        rank += 1

    return records


# ── Scrape one page ───────────────────────────────────────────────────────────

async def scrape(label: str, url: str) -> list[dict]:
    country  = label.replace("-", " ").title()
    all_rows = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        # Speed: block images/fonts (content is SSR)
        await page.route(
            "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,otf}",
            lambda r: r.abort()
        )

        print(f"\n[{label.upper()}] {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except PWTimeoutError:
            print("  timeout — continuing...")
        except Exception as e:
            print(f"  load error: {e} — continuing...")

        await page.wait_for_timeout(LOAD_WAIT)

        # Top 10
        print("  Extracting top-10 cards...")
        top10 = await extract_top10(page, country)
        print(f"  → {len(top10)} records")
        all_rows.extend(top10)

        top10_names_lower = {r["name"].lower() for r in top10}

        # Summary table
        print("  Extracting summary table...")
        rest = await extract_summary_table(
            page, country, top10_names_lower, rank_start=len(top10) + 1
        )
        print(f"  → {len(rest)} records")
        all_rows.extend(rest)

        await browser.close()

    return all_rows


# ── Export ────────────────────────────────────────────────────────────────────

def deduplicate(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = r["name"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    removed = len(rows) - len(out)
    if removed:
        print(f"  Deduped: removed {removed} → {len(out)} unique rows")
    return out


def save_csv(rows: list[dict], label: str) -> list[dict]:
    if not rows:
        print(f"  ⚠️  No data for {label}.")
        return []
    deduped = deduplicate(rows)
    path = os.path.join(OUTPUT_DIR, f"{label}_startups.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)
    print(f"  ✅ {path}  ({len(deduped)} rows)")
    return deduped

# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    all_sea = []

    for label, url in PAGES.items():
        rows    = await scrape(label, url)
        deduped = save_csv(rows, label)
        all_sea.extend(deduped)

    if all_sea:
        print(f"\n[COMBINED] {len(all_sea)} total rows across all SEA countries")
        combined = save_csv(all_sea, COMBINED_LABEL)
        # XLSX export removed to keep output paths consistent and avoid missing save_xlsx.



if __name__ == "__main__":
    asyncio.run(main())