"""
Scraper for growthlist.co - Ninja Tables / FooTable pagination
Handles the ConvertKit email popup that blocks clicks.

Install:
    pip install playwright
    playwright install chromium

Run:
    python scrape_growthlist.py
"""

import asyncio
import csv
import os
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

# ── Config ────────────────────────────────────────────────────────────────────
PAGES = {
    "brazil":    "https://growthlist.co/brazil-startups/",
    # "mexico":    "https://growthlist.co/mexico-startups/",
    # "colombia":  "https://growthlist.co/colombia-startups/",
    # "chile":     "https://growthlist.co/chile-startups/",
    # "argentina": "https://growthlist.co/argentina-startups/",
    # "peru":      "https://growthlist.co/peru-startups/",
}

OUTPUT_DIR = "."
HEADLESS   = True
CLICK_WAIT = 1_500  # ms after clicking Next

COLUMNS = [
    "Name", "Website", "Industry", "Country",
    "Funding_Amount_USD", "Funding_Type", "Last_Funding_Date",
]

SEL_ROWS    = "table.ninja_footable tbody tr, table.foo-table tbody tr"
SEL_NEXT_LI = "li.footable-page-nav[data-page='next']"
SEL_NEXT_A  = "li.footable-page-nav[data-page='next'] a.footable-page-link"

CLOSE_SELECTORS = [
    ".formkit-close",
    "button[data-formkit-close]",
    ".seva-close",
]


def clean_amount(raw: str) -> str:
    return raw.strip().lstrip("$").replace(",", "").strip()


async def dismiss_popup(page):
    for sel in CLOSE_SELECTORS:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click(timeout=3_000)
                await page.wait_for_timeout(500)
                print("  → Popup closed via close button.")
                return
        except Exception:
            pass

    try:
        removed = await page.evaluate("""() => {
            const selectors = [
                '.seva-overlay', '.formkit-overlay',
                '[data-object="overlay"]', '.seva-modal', '.formkit-modal',
            ];
            let removed = 0;
            selectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => {
                    el.style.display = 'none';
                    el.setAttribute('data-active', 'false');
                    removed++;
                });
            });
            document.body.style.overflow = '';
            document.body.style.position = '';
            return removed;
        }""")
        if removed:
            await page.wait_for_timeout(300)
            print(f"  → Popup hidden via JS ({removed} elements).")
    except Exception as e:
        print(f"  → JS popup dismiss failed: {e}")


async def popup_is_active(page) -> bool:
    try:
        return bool(await page.evaluate("""() => {
            const el = document.querySelector(
                '.seva-overlay, .formkit-overlay, [data-object="overlay"]'
            );
            if (!el) return false;
            return el.getAttribute('data-active') === 'true'
                || getComputedStyle(el).display !== 'none';
        }"""))
    except Exception:
        return False


async def extract_rows(page) -> list[dict]:
    results = []
    rows = await page.query_selector_all(SEL_ROWS)
    for row in rows:
        cells = await row.query_selector_all("td")
        texts = [(await c.inner_text()).strip() for c in cells]
        if len(texts) >= 7:
            results.append({
                "Name":               texts[0],
                "Website":            texts[1],
                "Industry":           texts[2],
                "Country":            texts[3],
                "Funding_Amount_USD": clean_amount(texts[4]),
                "Funding_Type":       texts[5],
                "Last_Funding_Date":  texts[6],
            })
    return results


async def next_disabled(page) -> bool:
    li = await page.query_selector(SEL_NEXT_LI)
    if li is None:
        return True
    cls = await li.get_attribute("class") or ""
    return "disabled" in cls


async def click_next(page) -> bool:
    if await popup_is_active(page):
        print("  → Popup detected, dismissing...")
        await dismiss_popup(page)

    try:
        clicked = await page.evaluate("""() => {
            const btn = document.querySelector(
                'li.footable-page-nav[data-page="next"] a.footable-page-link'
            );
            if (btn) { btn.click(); return true; }
            return false;
        }""")
        return bool(clicked)
    except Exception as e:
        print(f"  → JS click failed: {e}")
        return False


async def scrape(country: str, url: str) -> list[dict]:
    all_rows: list[dict] = []

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

        await page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}", lambda r: r.abort())

        print(f"\n[{country.upper()}] Loading {url} ...")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        except PWTimeoutError:
            print("  domcontentloaded timed out — continuing anyway...")
        except Exception as e:
            print(f"  goto error: {e} — continuing anyway...")

        print("  Waiting for table (6s)...")
        await page.wait_for_timeout(6_000)

        try:
            await page.wait_for_selector(SEL_ROWS, timeout=15_000)
            print("  ✓ Table rows found.")
        except PWTimeoutError:
            print("  ✗ No table rows. Saving debug screenshot...")
            await page.screenshot(path=f"{country}_debug.png", full_page=False)
            await browser.close()
            return all_rows

        if await popup_is_active(page):
            print("  → Popup on load, dismissing...")
            await dismiss_popup(page)

        page_num = 1
        while True:
            rows = await extract_rows(page)
            print(f"  Page {page_num}: {len(rows)} rows")
            all_rows.extend(rows)

            if await next_disabled(page):
                print("  → Last page reached.")
                break

            success = await click_next(page)
            if not success:
                print("  → Could not click Next.")
                break

            await page.wait_for_timeout(CLICK_WAIT)
            page_num += 1

            if page_num > 100:
                print("  → Safety cap (100 pages).")
                break

        await browser.close()

    return all_rows


def save_csv(rows: list[dict], country: str):
    path = os.path.join(OUTPUT_DIR, f"{country}_startups.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ {len(rows)} rows → {path}")


async def main():
    for country, url in PAGES.items():
        rows = await scrape(country, url)
        if rows:
            save_csv(rows, country)
        else:
            print(f"  ⚠️  No data for {country}.")

if __name__ == "__main__":
    asyncio.run(main())