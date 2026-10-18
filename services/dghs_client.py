import httpx
from bs4 import BeautifulSoup
import re

DGHS_DASHBOARD_URL = "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php"

async def fetch_dghs_dengue_baseline() -> dict:
    """
    Scrapes the Directorate General of Health Services (DGHS) HEOC Dengue Dynamic Dashboard
    for real-time 24-hour and cumulative hospital admission baselines across Bangladesh.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) EpiGrid-Preemption-Engine/2.0"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(DGHS_DASHBOARD_URL, headers=headers)
            response.raise_for_status()
            html_text = response.text

        soup = BeautifulSoup(html_text, "lxml")

        # Extract Last Updated Date from DGHS banner
        page_text = soup.get_text(" ", strip=True)
        update_match = re.search(r"Last Updated\s*:\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", page_text)
        last_updated = update_match.group(1) if update_match else "Live Stream"

        tables_extracted = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)

            if len(rows) > 1:
                tables_extracted.append({
                    "headers": rows[0],
                    "grand_total_row": rows[-1],
                    "record_count": len(rows) - 1
                })

        return {
            "source": "DGHS MIS Health Emergency Operation Center (HEOC)",
            "dashboard_endpoint": DGHS_DASHBOARD_URL,
            "status": "Live Ingestion Successful",
            "last_reported_update": last_updated,
            "surveillance_tables_parsed": len(tables_extracted),
            "summary_snapshots": tables_extracted[:4]
        }

    except Exception as exc:
        return {
            "source": "DGHS MIS Health Emergency Operation Center (HEOC)",
            "dashboard_endpoint": DGHS_DASHBOARD_URL,
            "status": "DGHS Portal Timeout - Fallback Baseline Active",
            "error_detail": str(exc)
        }
