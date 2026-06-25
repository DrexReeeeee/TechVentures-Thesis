"""
YC Latin America scraper — fixed version
Phase 1: Scroll directory, collect slugs + basic card data
Phase 2: Visit each YC profile for detailed data
Phase 3: Use Crunchbase API (unofficial) for funding data

Install:
    pip install playwright requests
    playwright install chromium

Run:
    python scrape_yc_latam.py
"""

import asyncio
import csv
import json
import time
import re
import requests
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

# ── Config ────────────────────────────────────────────────────────────────────
DIRECTORY_URL = "https://www.ycombinator.com/companies/?regions=Latin%20America"
OUTPUT_FILE   = "yc_latam_companies.csv"
HEADLESS      = True
SCROLL_PAUSE  = 2_500
MAX_SAME      = 5
PROFILE_DELAY = 1_500   # ms between YC profile requests
CB_DELAY      = 2_000   # ms between Crunchbase requests

COLUMNS = [
    "Name", "Slug", "YC_Batch", "Status", "Founded_Year", "Team_Size",
    "Location", "Short_Description", "Long_Description",
    "Industries", "Website",
    "LinkedIn", "Twitter", "Crunchbase", "GitHub",
    "Founders", "Founder_LinkedIns",
    "YC_URL",
    "CB_Total_Funding_USD", "CB_Last_Funding_Type", "CB_Last_Funding_Date",
    "CB_Num_Funding_Rounds", "CB_Investors",
    "CB_Valuation_USD", "CB_IPO_Date", "CB_Acquired_By",
]


# ── Phase 1: collect slugs from directory ────────────────────────────────────
async def collect_slugs(page) -> list[str]:
    same_count = 0
    last_count = 0
    scroll_n   = 0
    slugs: set[str] = set()

    while same_count < MAX_SAME:
        found = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('a[href^="/companies/"]'))
                .map(a => a.getAttribute('href'))
                .filter(h => {
                    if (!h) return false;
                    const parts = h.replace('/companies/', '').split('/');
                    // Must be a single slug, no sub-paths like /industry/, /jobs
                    return parts.length === 1 && parts[0].length > 0
                        && !h.includes('?') && h !== '/companies';
                });
        }""")
        for h in found:
            slug = h.replace("/companies/", "")
            slugs.add(slug)

        scroll_n += 1
        print(f"  Scroll {scroll_n}: {len(slugs)} companies")

        if len(slugs) == last_count:
            same_count += 1
        else:
            same_count = 0
            last_count = len(slugs)

        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(SCROLL_PAUSE)

    return sorted(slugs)


# ── Phase 2: scrape YC profile ───────────────────────────────────────────────
async def scrape_yc_profile(page, slug: str) -> dict:
    url = f"https://www.ycombinator.com/companies/{slug}"
    data = {col: "" for col in COLUMNS}
    data["Slug"]   = slug
    data["YC_URL"] = url

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(2_000)
    except Exception as e:
        print(f"    ⚠ Load error: {e}")
        return data

    try:
        result = await page.evaluate("""() => {
            // ── Name: the <h1> on the page ──────────────────────────────
            const h1 = document.querySelector('h1');
            const name = h1 ? h1.innerText.trim() : '';

            // ── Short tagline: the <p> immediately after the name / under h1 ──
            // YC renders it as a styled paragraph right below the company name
            // We want the first <p> that looks like a tagline (short, not a label)
            let shortDesc = '';
            const allP = Array.from(document.querySelectorAll('p'));
            for (const p of allP) {
                const t = p.innerText.trim();
                if (t.length > 10 && t.length < 300 && !t.includes('\\n')) {
                    shortDesc = t;
                    break;
                }
            }

            // ── Long description ─────────────────────────────────────────
            let longDesc = '';
            for (const p of allP) {
                const t = p.innerText.trim();
                if (t.length > 100) {
                    longDesc = t;
                    break;
                }
            }

            // ── Batch: the link with text like "Winter 2016" or "W24" ──
            const batchEl = document.querySelector('a[href*="batch="]');
            const batch   = batchEl ? batchEl.innerText.trim() : '';

            // ── Status, Founded, Team Size from structured sidebar ────────
            // YC renders these in a definition-list style: label then value
            const bodyText = document.body.innerText;

            const statusRe = bodyText.match(/\\b(Active|Acquired|Public|Inactive)\\b/);
            const status   = statusRe ? statusRe[1] : '';

            const foundedRe = bodyText.match(/Founded[:\\s]+(\\d{4})/i);
            const founded   = foundedRe ? foundedRe[1] : '';

            const teamRe  = bodyText.match(/Team Size[:\\s]+(\\d[\\d,]*)/i);
            const teamSize = teamRe ? teamRe[1] : '';

            // ── Location from meta description ────────────────────────────
            const metaEl  = document.querySelector('meta[name="description"]');
            const metaStr = metaEl ? metaEl.getAttribute('content') : '';
            const locRe   = metaStr.match(/based in ([^.]+)\\./);
            const location = locRe ? locRe[1].trim() : '';

            // ── Industries ────────────────────────────────────────────────
            const indLinks = Array.from(
                document.querySelectorAll('a[href*="/companies/industry/"]')
            );
            const industries = [...new Set(indLinks.map(a => a.innerText.trim()))]
                .filter(Boolean).join(', ');

            // ── External links: website, social ──────────────────────────
            // Exclude YC internal links and image CDN links
            const extLinks = Array.from(document.querySelectorAll('a[href^="http"]'))
                .filter(a =>
                    !a.href.includes('ycombinator.com') &&
                    !a.href.includes('bookface-images') &&
                    !a.href.includes('startupschool.org')
                );

            const findLink = (keywords) =>
                extLinks.find(a => keywords.some(k => a.href.toLowerCase().includes(k)))?.href || '';

            const website    = extLinks.find(a =>
                !['linkedin','twitter','x.com','crunchbase','github','facebook','instagram']
                    .some(k => a.href.toLowerCase().includes(k)))?.href || '';
            const linkedin   = findLink(['linkedin.com/company']);
            const twitter    = findLink(['twitter.com/', 'x.com/']);
            const crunchbase = findLink(['crunchbase.com/organization']);
            const github     = findLink(['github.com/']);

            // ── Founders: find personal LinkedIn links ────────────────────
            const founderLinks = Array.from(
                document.querySelectorAll('a[href*="linkedin.com/in/"]')
            );
            const founderLinkedIn = [...new Set(founderLinks.map(a => a.href))];

            // For founder names: look at text nodes near the LinkedIn link
            const founderNames = [];
            founderLinks.forEach(el => {
                // Walk up DOM to find a container with the founder's name
                let parent = el.parentElement;
                for (let i = 0; i < 5; i++) {
                    if (!parent) break;
                    // Look for a heading or strong text in this container
                    const nameEl = parent.querySelector('h3, h4, strong, b, [class*="name"]');
                    if (nameEl) {
                        const n = nameEl.innerText.split('\\n')[0].trim();
                        if (n && n.length < 60 && !founderNames.includes(n)) {
                            founderNames.push(n);
                            break;
                        }
                    }
                    parent = parent.parentElement;
                }
            });

            return {
                name, shortDesc, longDesc, batch, status, founded, teamSize,
                location, industries, website, linkedin, twitter, crunchbase,
                github,
                founderNames: founderNames.join(' | '),
                founderLinkedIns: founderLinkedIn.join(' | '),
            };
        }""")

        data["Name"]               = result.get("name", "")
        data["YC_Batch"]           = result.get("batch", "")
        data["Status"]             = result.get("status", "")
        data["Founded_Year"]       = result.get("founded", "")
        data["Team_Size"]          = result.get("teamSize", "")
        data["Location"]           = result.get("location", "")
        data["Short_Description"]  = result.get("shortDesc", "")
        data["Long_Description"]   = result.get("longDesc", "")
        data["Industries"]         = result.get("industries", "")
        data["Website"]            = result.get("website", "")
        data["LinkedIn"]           = result.get("linkedin", "")
        data["Twitter"]            = result.get("twitter", "")
        data["Crunchbase"]         = result.get("crunchbase", "")
        data["GitHub"]             = result.get("github", "")
        data["Founders"]           = result.get("founderNames", "")
        data["Founder_LinkedIns"]  = result.get("founderLinkedIns", "")

    except Exception as e:
        print(f"    ⚠ Extraction error: {e}")

    return data


# ── Phase 3: Crunchbase via their unofficial API ─────────────────────────────
CB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "x-cb-user-key": "",   # empty — works for public data
}

def get_cb_slug(cb_url: str) -> str:
    """Extract slug from https://www.crunchbase.com/organization/rappi → rappi"""
    m = re.search(r"/organization/([^/?#]+)", cb_url)
    return m.group(1) if m else ""


async def scrape_crunchbase(page, cb_url: str) -> dict:
    """Scrape Crunchbase page via Playwright — handles JS rendering."""
    cb = {k: "" for k in [
        "CB_Total_Funding_USD", "CB_Last_Funding_Type", "CB_Last_Funding_Date",
        "CB_Num_Funding_Rounds", "CB_Investors",
        "CB_Valuation_USD", "CB_IPO_Date", "CB_Acquired_By",
    ]}

    if not cb_url:
        return cb

    try:
        await page.goto(cb_url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(4_000)  # wait for React hydration
    except Exception as e:
        print(f"      ⚠ CB load error: {e}")
        return cb

    try:
        result = await page.evaluate("""() => {
            const text = document.body.innerText;

            // Helper: find value after a label
            const after = (label, maxLen=60) => {
                const re = new RegExp(label + '[\\\\s\\\\S]{0,' + maxLen + '?}([\\\\$\\\\w][^\\\\n]+)', 'i');
                const m = text.match(re);
                return m ? m[1].trim() : '';
            };

            // Total Funding
            const fundRe = text.match(/Total Funding Amount[\\s\\S]{0,20}?(\\$[\\d.,]+\\s*[BMK]?)/i)
                        || text.match(/(\\$[\\d.,]+\\s*[BMK])\\s*Total Funding/i);
            const totalFunding = fundRe ? fundRe[1].trim() : '';

            // Number of funding rounds
            const roundsRe = text.match(/(\\d+)\\s+Funding Rounds?/i);
            const numRounds = roundsRe ? roundsRe[1] : '';

            // Last funding type
            const typeRe = text.match(/Last Funding Type[\\s\\S]{0,20}?(Series [A-Z]+\\+?|Seed|Pre-Seed|Angel|Convertible Note|Debt Financing|Grant|IPO|Post-IPO|Acquired|Corporate Round)/i);
            const lastType = typeRe ? typeRe[1].trim() : '';

            // Last funding date — look for "Announced Date" or "Last Funding Date"
            const dateRe = text.match(/(?:Last Funding Date|Announced Date)[\\s\\S]{0,30}?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+\\d+,?\\s*\\d{4})/i);
            const lastDate = dateRe ? dateRe[1].trim() : '';

            // Investors — collect links in the funding/investors section
            const invEls = Array.from(document.querySelectorAll(
                'a[href*="/organization/"], a[href*="/person/"]'
            )).filter(a => {
                const h = a.href;
                return !h.includes('crunchbase.com/organization/' + window.location.pathname.split('/').pop());
            });
            const investors = [...new Set(invEls.map(a => a.innerText.trim()))]
                .filter(t => t.length > 1 && t.length < 80)
                .slice(0, 25).join(', ');

            // Valuation
            const valRe = text.match(/(?:Valuation|Post-Money Valuation)[\\s\\S]{0,20}?(\\$[\\d.,]+\\s*[BMK]?)/i);
            const valuation = valRe ? valRe[1].trim() : '';

            // IPO date
            const ipoRe = text.match(/IPO(?:\\s+Date)?[\\s\\S]{0,30}?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\s+\\d+,?\\s*\\d{4})/i);
            const ipoDate = ipoRe ? ipoRe[1].trim() : '';

            // Acquired by
            const acqRe = text.match(/Acquired by\\s+([A-Z][\\w\\s&.,]+?)(?:\\s+on|\\s+for|\\n|$)/i);
            const acquiredBy = acqRe ? acqRe[1].trim() : '';

            return {
                totalFunding, numRounds, lastType, lastDate,
                investors, valuation, ipoDate, acquiredBy
            };
        }""")

        cb["CB_Total_Funding_USD"]  = result.get("totalFunding", "")
        cb["CB_Last_Funding_Type"]  = result.get("lastType", "")
        cb["CB_Last_Funding_Date"]  = result.get("lastDate", "")
        cb["CB_Num_Funding_Rounds"] = result.get("numRounds", "")
        cb["CB_Investors"]          = result.get("investors", "")
        cb["CB_Valuation_USD"]      = result.get("valuation", "")
        cb["CB_IPO_Date"]           = result.get("ipoDate", "")
        cb["CB_Acquired_By"]        = result.get("acquiredBy", "")

    except Exception as e:
        print(f"      ⚠ CB extraction error: {e}")

    return cb


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
        )

        block = lambda r: r.abort()

        # ── Phase 1 ───────────────────────────────────────────────────────
        print("=" * 55)
        print("Phase 1: Collecting company slugs from YC directory")
        print("=" * 55)
        dir_page = await context.new_page()
        await dir_page.route("**/*.{png,jpg,jpeg,gif,woff,woff2,ttf,mp4}", block)

        try:
            await dir_page.goto(DIRECTORY_URL, wait_until="domcontentloaded", timeout=60_000)
        except PWTimeoutError:
            print("  Timeout — continuing...")

        await dir_page.wait_for_timeout(5_000)
        try:
            await dir_page.wait_for_selector("a[href^='/companies/']", timeout=20_000)
        except PWTimeoutError:
            print("  No links found. Exiting.")
            await browser.close()
            return

        slugs = await collect_slugs(dir_page)
        print(f"\n  ✓ {len(slugs)} slugs collected\n")
        await dir_page.close()

        # ── Phase 2 + 3 ───────────────────────────────────────────────────
        print("=" * 55)
        print("Phase 2+3: YC profiles + Crunchbase data")
        print("=" * 55)

        yc_page = await context.new_page()
        cb_page = await context.new_page()
        await yc_page.route("**/*.{png,jpg,jpeg,gif,woff,woff2,ttf,mp4}", block)
        await cb_page.route("**/*.{png,jpg,jpeg,gif,woff,woff2,ttf,mp4}", block)

        all_data = []

        for i, slug in enumerate(slugs, 1):
            print(f"\n  [{i}/{len(slugs)}] {slug}")

            # YC profile
            data = await scrape_yc_profile(yc_page, slug)
            print(f"    Name: {data['Name'] or '(not found)'} | Batch: {data['YC_Batch']}")
            await asyncio.sleep(PROFILE_DELAY / 1000)

            # Crunchbase
            cb_url = data.get("Crunchbase", "")
            if cb_url:
                print(f"    CB: {cb_url}")
                cb_data = await scrape_crunchbase(cb_page, cb_url)
                data.update(cb_data)
                funding_str = cb_data.get('CB_Total_Funding_USD', '') or '(no data)'
                print(f"    Funding: {funding_str} | Rounds: {cb_data.get('CB_Num_Funding_Rounds','')}")
                await asyncio.sleep(CB_DELAY / 1000)
            else:
                print(f"    No Crunchbase link")

            all_data.append(data)

        await yc_page.close()
        await cb_page.close()
        await browser.close()

        # ── Save ──────────────────────────────────────────────────────────
        if all_data:
            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(all_data)
            print(f"\n✅ {len(all_data)} companies → {OUTPUT_FILE}")
        else:
            print("⚠️  No data.")


if __name__ == "__main__":
    asyncio.run(main())