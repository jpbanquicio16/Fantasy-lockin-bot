"""Official NBA injury reports: discovery, an immutable PDF archive, and a table parser.

Reports are PDFs at ``ak-static.cms.nba.com/referee/injury/Injury-Report_<ET date>_<hh>_<mm><AM|PM>.pdf``,
one per 15-minute slot on game days (the report page lists them; that page answers 403
to automated requests, so slots are constructed and tried in turn). The file name's time
is US Eastern and becomes the publication instant (knowledge time) in UTC.

Layout (fixtures in tests/fixtures/injury_reports, captured 6 Nov 2026): columns Game Date,
Game Time, Matchup, Team, Player Name, Current Status, Reason at fixed x positions; game,
matchup and team appear only on the first row of their block and are carried down, across
page breaks too; names lose their spaces ("ButlerIII,Jimmy"); a reason may wrap onto the
lines above and below its row; a team that has not reported shows "NOTYETSUBMITTED".
Every row is checked; an unknown status fails the parse instead of guessing.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from lockin_core.config import ET, UTC, Paths, now_utc
from lockin_core.ingest.raw import HttpClient

BASE = "https://ak-static.cms.nba.com"
STATUSES = ("Available", "Probable", "Questionable", "Doubtful", "Out")
TEAM_NAMES = {
    "AtlantaHawks": "ATL", "BostonCeltics": "BOS", "BrooklynNets": "BKN", "CharlotteHornets": "CHA",
    "ChicagoBulls": "CHI", "ClevelandCavaliers": "CLE", "DallasMavericks": "DAL", "DenverNuggets": "DEN",
    "DetroitPistons": "DET", "GoldenStateWarriors": "GSW", "HoustonRockets": "HOU", "IndianaPacers": "IND",
    "LAClippers": "LAC", "LosAngelesClippers": "LAC", "LosAngelesLakers": "LAL", "MemphisGrizzlies": "MEM",
    "MiamiHeat": "MIA", "MilwaukeeBucks": "MIL", "MinnesotaTimberwolves": "MIN",
    "NewOrleansPelicans": "NOP", "NewYorkKnicks": "NYK", "OklahomaCityThunder": "OKC",
    "OrlandoMagic": "ORL", "Philadelphia76ers": "PHI", "PhoenixSuns": "PHX",
    "PortlandTrailBlazers": "POR", "SacramentoKings": "SAC", "SanAntonioSpurs": "SAS",
    "TorontoRaptors": "TOR", "UtahJazz": "UTA", "WashingtonWizards": "WAS",
}
HEADERS = ("GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason")
DEFAULT_BOUNDS = (23.0, 120.0, 200.0, 264.0, 425.0, 586.0, 666.0)
SUFFIX = re.compile(r"(?<=[a-z.'])(Jr\.?|Sr\.?|III|II|IV|V)$")
NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


class InjuryReportParseError(ValueError):
    pass


# --- names -----------------------------------------------------------------------
def name_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def report_name_key(raw: str) -> str:
    """"ButlerIII,Jimmy" -> "jimmybutler"; "CenacJr.,Christopher" -> "christophercenac"."""
    parts = [x for x in raw.split(",") if x]
    if len(parts) >= 3 and name_key(parts[1]) in NAME_SUFFIX_TOKENS:  # "Udeh,Jr.,Ernest"
        parts = [parts[0], parts[2]]
    last, first = (parts + [""])[:2]
    last = SUFFIX.sub("", last.strip())
    return name_key(first + last)


def sleeper_name_key(full_name: str) -> str:
    toks = [t for t in re.split(r"\s+", full_name.strip()) if t]
    toks = [t for t in toks if name_key(t) not in NAME_SUFFIX_TOKENS] or toks
    return name_key("".join(toks))


# --- slots and addresses -------------------------------------------------------------
def slot_name(ts_et: datetime) -> str:
    h12 = ts_et.hour % 12 or 12
    return f"Injury-Report_{ts_et:%Y-%m-%d}_{h12:02d}_{ts_et.minute:02d}{'AM' if ts_et.hour < 12 else 'PM'}.pdf"


def parse_slot_name(name: str) -> datetime:
    m = re.match(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf$", name)
    if not m:
        raise ValueError(f"not a report file name: {name}")
    d, hh, mm, ap = m.groups()
    h = int(hh) % 12 + (12 if ap == "PM" else 0)
    local = datetime.fromisoformat(f"{d}T{h:02d}:{mm}:00").replace(tzinfo=ET)
    return local.astimezone(UTC)


def slots_between(start_utc: datetime, end_utc: datetime) -> list[datetime]:
    """Every 15-minute ET slot in (start, end], as ET datetimes."""
    t = start_utc.astimezone(ET)
    t = t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0) + timedelta(minutes=15)
    out = []
    while t.astimezone(UTC) <= end_utc:
        out.append(t)
        t = (t.astimezone(UTC) + timedelta(minutes=15)).astimezone(ET)
    return out


# --- parsing ---------------------------------------------------------------------
@dataclass
class ReportRow:
    game_date: date | None
    game_time: str | None
    matchup: str | None
    team: str | None
    player_raw: str | None
    player_key: str | None
    status: str | None
    reason: str
    not_submitted: bool
    page: int


def _lines(words: list[dict], tol: float = 2.5) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if out and abs(out[-1][0]["top"] - w["top"]) <= tol:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _column(x0: float, bounds: tuple[float, ...]) -> int:
    col = 0
    for i, b in enumerate(bounds):
        if x0 + 3 >= b:
            col = i
    return col


def parse_pdf(data: bytes) -> list[ReportRow]:
    import pdfplumber

    rows: list[ReportRow] = []
    bounds = DEFAULT_BOUNDS
    carry: dict[str, object] = {"date": None, "time": None, "matchup": None, "team": None}
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for pno, page in enumerate(pdf.pages):
            words = page.extract_words()
            hdr = {w["text"]: w["x0"] for w in words if w["text"] in HEADERS}
            if len(hdr) == len(HEADERS):
                bounds = tuple(float(hdr[h]) - 1 for h in HEADERS)
            page_rows: list[tuple[float, ReportRow]] = []
            fragments: list[tuple[float, str]] = []
            for line in _lines(words):
                texts = [w["text"] for w in line]
                if texts[:2] == ["Injury", "Report:"] or "GameDate" in texts or (
                        len(texts) == 1 and re.fullmatch(r"Page\d+of\d+", texts[0])):
                    continue
                cells: dict[int, list[str]] = {}
                for w in line:
                    cells.setdefault(_column(w["x0"], bounds), []).append(w["text"])
                cell = {k: " ".join(v) for k, v in cells.items()}
                top = line[0]["top"]
                if 0 in cell:
                    carry["date"] = datetime.strptime(cell[0], "%m/%d/%Y").date()
                if 1 in cell:
                    carry["time"] = cell[1]
                if 2 in cell:
                    carry["matchup"] = cell[2]
                if 3 in cell:
                    name = cell[3].replace(" ", "")
                    if name not in TEAM_NAMES:
                        raise InjuryReportParseError(f"page {pno + 1}: unknown team {cell[3]!r}")
                    carry["team"] = TEAM_NAMES[name]
                reason = cell.get(6, "")
                status = cell.get(5)
                not_sub = "NOTYETSUBMITTED" in reason.replace(" ", "") or (
                    status is not None and "NOTYETSUBMITTED" in status.replace(" ", ""))
                base = dict(game_date=carry["date"], game_time=carry["time"], matchup=carry["matchup"],
                            team=carry["team"], page=pno + 1)
                if not_sub:
                    page_rows.append((top, ReportRow(player_raw=None, player_key=None, status=None, reason="",
                                                     not_submitted=True, **base)))
                elif 4 in cell or status is not None:
                    if status not in STATUSES or 4 not in cell:
                        raise InjuryReportParseError(
                            f"page {pno + 1}: row {texts} has no valid status (expected one of {STATUSES})")
                    raw = cell[4].replace(" ", "")
                    page_rows.append((top, ReportRow(player_raw=raw, player_key=report_name_key(raw),
                                                     status=status, reason=reason, not_submitted=False,
                                                     **base)))
                elif set(cell) == {6}:
                    fragments.append((top, reason))
                elif set(cell) - {0, 1, 2, 3}:
                    raise InjuryReportParseError(f"page {pno + 1}: unrecognised line {texts}")
            players = [(t, r) for t, r in page_rows if not r.not_submitted]
            parts: dict[int, list[tuple[float, str]]] = {}
            for ft, txt in fragments:
                if not players:
                    continue
                j = min(range(len(players)), key=lambda i: abs(players[i][0] - ft))
                parts.setdefault(j, []).append((ft, txt))
            for j, frs in parts.items():
                t, r = players[j]
                pieces = sorted(frs + ([(t, r.reason)] if r.reason else []))
                r.reason = "".join(p for _, p in pieces)
            rows.extend(r for _, r in page_rows)
    for r in rows:
        if r.team is None or r.game_date is None:
            raise InjuryReportParseError(f"row without game or team: {r}")
    return rows


def game_time_et(game_date: date, game_time: str | None) -> datetime | None:
    """"07:30(ET)" on a game date -> the ET datetime. The report omits AM/PM; every NBA tip is
    noon or later ET, so hours 1-11 are read as PM."""
    if not game_time:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})", game_time)
    if not m:
        return None
    h, mm = int(m.group(1)), int(m.group(2))
    h = h if h == 12 else h + 12
    return datetime(game_date.year, game_date.month, game_date.day, h, mm, tzinfo=ET)


# --- archive and client ----------------------------------------------------------
class InjuryReportArchive:
    """``data/raw/nba_injury/<ET date>/<file>.pdf`` plus a JSON sidecar; never overwritten."""

    def __init__(self, paths: Paths):
        self.root = paths.raw / "nba_injury"

    def path(self, name: str) -> Path:
        return self.root / name[len("Injury-Report_"):len("Injury-Report_") + 10] / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def put(self, name: str, data: bytes, fetched_at: datetime) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
            meta = {"name": name, "published_at": parse_slot_name(name).isoformat(),
                    "fetched_at": fetched_at.isoformat(), "sha256": hashlib.sha256(data).hexdigest()}
            p.with_suffix(".json").write_text(json.dumps(meta))
        return p

    def missing_marker(self, name: str) -> Path:
        return self.path(name).with_suffix(".missing")

    def all(self) -> list[Path]:
        return sorted(self.root.glob("*/*.pdf")) if self.root.exists() else []

    def latest_name(self) -> str | None:
        files = self.all()
        return max(files, key=lambda p: parse_slot_name(p.name)).name if files else None

    def meta(self, p: Path) -> dict:
        m = p.with_suffix(".json")
        return json.loads(m.read_text()) if m.exists() else {
            "name": p.name, "published_at": parse_slot_name(p.name).isoformat(), "fetched_at": None}


class InjuryReportClient:
    def __init__(self, paths: Paths | None = None, http: HttpClient | None = None):
        self.paths = (paths or Paths()).ensure()
        self.archive = InjuryReportArchive(self.paths)
        self.http = http or HttpClient(BASE, min_interval=1.0, retries=2, timeout=20.0)

    def fetch_slot(self, ts_et: datetime) -> Path | None:
        name = slot_name(ts_et)
        if self.archive.has(name):
            return self.archive.path(name)
        status, body, ctype = self.http.get_bytes(f"/referee/injury/{name}")
        if status == 200 and body[:4] == b"%PDF":
            return self.archive.put(name, body, now_utc())
        return None

    def discover(self, now: datetime | None = None, lookback_hours: float = 4.0,
                 max_requests: int = 24) -> list[Path]:
        """Fetch every slot published since the newest archived report (at most
        ``lookback_hours`` back), newest last."""
        now = now or now_utc()
        start = now - timedelta(hours=lookback_hours)
        latest = self.archive.latest_name()
        if latest is not None:
            start = max(start, parse_slot_name(latest))
        found = []
        for ts in slots_between(start, now)[-max_requests:]:
            p = self.fetch_slot(ts)
            if p is not None:
                found.append(p)
        return found
