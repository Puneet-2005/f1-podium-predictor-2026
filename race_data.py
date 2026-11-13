"""Read the bundled race data or refresh it from Formula 1's results pages."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from lxml import html

SITE = "https://www.formula1.com"
INDEX = "/en/results/2026/races"
RACE_LINK = re.compile(rf"{INDEX}/\d+/([a-z0-9-]+)/race-result")
DATA_PATH = Path(__file__).resolve().parent / "data" / "2026.json"


def fetch(path):
    request = Request(SITE + path, headers={"User-Agent": "Mozilla/5.0 F1 race analysis"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=15) as response:
                return html.fromstring(response.read())
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def race_links(index):
    completed = {}
    for row in index.xpath("//table[1]//tbody/tr"):
        for path in row.xpath("./td[1]//a/@href"):
            match = RACE_LINK.fullmatch(path)
            if match:
                completed[match[1]] = path
    if not completed:
        raise ValueError("Couldn't read the completed races from Formula 1's results page.")
    all_races = {
        match[1]: path
        for path in index.xpath("//a/@href")
        if (match := RACE_LINK.fullmatch(path))
    }
    return completed, all_races


def table_rows(page, expected_column):
    for table in page.xpath("//table"):
        headings = [cell.text_content().strip() for cell in table.xpath(".//th")]
        if {"Pos.", "No.", "Driver", "Team", expected_column}.issubset(headings):
            rows = []
            for tr in table.xpath(".//tbody/tr"):
                cells = tr.xpath("./td")
                if len(cells) != len(headings):
                    continue
                row = dict(zip(headings, (cell.text_content().strip() for cell in cells)))
                if not row["No."].isdigit():
                    continue
                driver = cells[headings.index("Driver")]
                short_name = driver.xpath(".//span[contains(concat(' ', @class, ' '), ' md:hidden ')]")
                if short_name:
                    row["Driver"] = short_name[-1].text_content().strip()
                rows.append(row)
            return rows
    return []


def lap_seconds(value):
    minutes, separator, seconds = value.partition(":")
    if not separator:
        return None
    return 60 * int(minutes) + float(seconds)


def parse_race(slug, path, pages):
    grid = table_rows(pages["starting-grid"], "Time")
    if not grid:
        raise ValueError(f"The starting grid for {slug} is unavailable. Run after it is published.")
    qualifying = table_rows(pages["qualifying"], "Q1")
    if not qualifying:
        raise ValueError(f"Couldn't read qualifying results for {slug}.")
    q1 = {row["No."]: lap_seconds(row["Q1"]) for row in qualifying}
    fastest = min((lap for lap in q1.values() if lap is not None), default=None)
    # Preserve the published order of any entries without a numbered grid slot.
    ordered_grid = sorted(grid, key=lambda row: int(row["Pos."]) if row["Pos."].isdigit() else 999)
    entries = []
    for position, row in enumerate(ordered_grid, 1):
        lap = q1.get(row["No."])
        entries.append({
            "number": row["No."],
            "driver": row["Driver"],
            "team": row["Team"],
            "grid": position,
            "q1_gap_pct": round(100 * (lap / fastest - 1), 4) if lap is not None else None,
        })
    result = table_rows(pages["race-result"], "Time / Retired") if "race-result" in pages else []
    if "race-result" in pages and not result:
        raise ValueError(f"Couldn't read the published classification for {slug}.")
    return {
        "race": slug,
        "source": SITE + path,
        "entries": entries,
        "results": {row["No."]: rank for rank, row in enumerate(result, 1)},
    }


def load_data(path=DATA_PATH):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def update_data(path, target=None):
    completed, all_races = race_links(fetch(INDEX))
    if target and target not in all_races:
        raise ValueError(f"Unknown 2026 race: {target}. Options: {', '.join(all_races)}")
    previous = {race["race"]: race for race in load_data(path)} if path.exists() else {}
    pending = [(slug, link) for slug, link in completed.items()
               if not previous.get(slug, {}).get("results")]
    if target and target not in completed:
        pending.append((target, all_races[target]))

    def download(race):
        slug, link = race
        base = link.rsplit("/", 1)[0]
        pages = {key: fetch(f"{base}/{key}") for key in ("starting-grid", "qualifying")}
        if slug in completed:
            pages["race-result"] = fetch(link)
        return slug, parse_race(slug, link, pages)

    with ThreadPoolExecutor(max_workers=4) as pool:
        previous.update(pool.map(download, pending))
    races = [previous[slug] for slug in completed]
    if target and target not in completed:
        races.append(previous[target])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(races, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return races
