# Fantasy Lock-In Assistant

A local Python service for Sleeper NBA Lock-In: lineup recommendations, lock/pass/wait decisions, an eligibility ledger and deadline notifications. All basketball actions happen in Sleeper; this application only records your actions.

## Setup

Python 3.11+ is required; the checked environment uses Python 3.13. Install from the repository directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e .
cp configs/league.example.yaml configs/league.yaml
```

Set `league_id` and either `my_roster_id` or `username`. Confirm the scoring switches against your league before enabling exact scoring. Existing local configuration and data are preserved.

```sh
.venv/bin/lockin league-tests --help
.venv/bin/lockin ingest
.venv/bin/lockin reconcile
.venv/bin/lockin fit
.venv/bin/lockin app
```

The dashboard listens on `127.0.0.1:8501`. Start `lockin run` in a separate process to collect lineups, detect finals, generate advice and check deadlines. It must remain running; the dashboard alone is not the event engine.

## Recording history

The API cannot establish lock state. Open **Confirm open scores before the first recommendation** in the dashboard, load the ledger, and confirm whether each score is still unlocked. Unknown tip history and unknown lock state block advice. You can correct tip history, record a completed score during a feed outage, edit scores, record locks/passes and set availability/minutes overrides.

```sh
lockin state
lockin confirm-tip '2026-11-06T00:30:00+00:00' 'player1,player2,-' --roster 1
lockin record not-locked 'Player Name'
lockin record lock 'Player Name' --slot-index 2 --score 31.5
lockin record pass 'Player Name'
lockin record opp-locked 120.5
lockin override 'Player Name' --p-play 0.6 --hours 8
```

Slot indices are zero based and distinguish duplicate C/UTIL slots. Historical lock corrections require `--game GAME_ID --slot-index N --score FP --at ISO_TIME`. To confirm a passed/unused historical score, use `record pass` or `record not-locked` with `--game`. Corrections change the local ledger, never Sleeper. The dashboard can retract an incorrect local lock record.

## Notifications and deployment

Copy `configs/alerts.example.yaml` to `configs/alerts.yaml`, or supply `LOCKIN_ALERT_URLS` as a JSON array of Apprise URLs. Keep credentials out of version control. `LOCKIN_HEARTBEAT_URL` supplies an external heartbeat URL. Without a configured push channel, notifications are logged locally; phone delivery is **not** verified.

Deadlines persist in SQLite and are checked even when a feed or model fails. Expired advice is replaced by the saved default, failed sends are retried, and missed alerts are recorded. The service cannot guarantee delivery while the host or network is down.

`deploy/` contains systemd examples for an always-on Linux host. Use the host's Tailscale address with `lockin app --address IP`; wildcard and public binds are rejected. These templates are not installed or started automatically. Run `lockin check-host --help` and the host checks on the intended machine, then verify phone access and notifications there.

Nightly backups include SQLite, ledger exports, tables and artifacts. Set `LOCKIN_BACKUP_DIR` to a separate mounted backup location; the default `data/backups` is on the same disk. A manual `lockin backup DESTINATION` exports the database and ledger. Restore only with the service stopped, and retain model/artifact directories for replay.

## Evaluation and audit

```sh
.venv/bin/python -m pytest -q
lockin toy --weeks 200 --players 5 --scenarios 20000
lockin toy --weeks 2 --players 3 --scenarios 64 --rollout 1
lockin bench --scenarios 2000
lockin replay --season 2025 --weeks 2 --rollout-weeks 2
lockin shadow
lockin audit --limit 5
```

Recommendations save their complete frozen model inputs as compressed, typed JSON. `audit` recreates the decision at the original scenario count without fetching current data, and compares canonical advice and numerical evidence. Runtime, generated IDs, current code metadata and the wall-clock limitation flag are excluded. Nightly auditing samples approximately 5% of the preceding day's recommendations.

Shadow scoring uses actual box scores, archived injury information and week-end lineup history. Missing observations remain pending. The report compares recorded actions and replayed counterfactuals with whole-week bootstrap intervals. You can record a separately verified outcome with:

```sh
lockin outcome REC_ID --action 'locked in G' --matched --realised 310 --recommended 314
```

`--recommended` means a chronological counterfactual week total, not a hindsight maximum. Synthetic performance and replay are not evidence of benefit in your league. The design's 100,000-week invariant test, 500-case optimality gate, host runtime target and prospective logging gates have not been certified by this implementation pass.

Model fits retain versioned parameters and residual libraries. The weekly service fit stages a candidate without activating it. After evaluation, `lockin activate-model VERSION` activates it or rolls back; explicit `lockin fit` activates the manually requested fit.

See [the implementation audit](docs/implementation-audit.md) for design coverage, differences and remaining external checks.
