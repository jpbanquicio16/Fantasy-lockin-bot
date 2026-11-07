# Design implementation audit — 6 November 2026

Compared against **Sleeper Lock-In Lineup and Lock Assistant — Technical Design and Architecture** supplied with the task.

## Existing implementation retained

Scoring and reconciliation; Sleeper ingestion with raw archives; ESPN schedule/status fallback; official injury PDF parsing; typed Parquet tables; projections and status calibration; stopping tables, assignment, base policy v2, exact toy DP; candidate screening, common random numbers, adaptive simulation, conditional advice and uncertainty bands; CLI, league probes and host checks.

## Added or repaired

| Design area | Implementation |
| --- | --- |
| §7 eligibility | Tip history controls eligibility; current matchup points do not establish eligibility or a lock. Unknown tip/lock state blocks live advice. Benched-at-tip games and observed lineup churn cannot create lockable entries. |
| §21 corrections | Append-only tip revisions and retractions; historical reads respect when corrections were entered. Manual locks retain their recorded slot. |
| §24 legality | Shared candidate/simulator legality checks, duplicate-player/slot rejection, frozen-slot checks, exact deadline boundary checks. |
| §7 information access | Time-gated score/status access; simultaneous finals revealed before decisions. |
| §14/20 alerts | Deadline checks survive feed failures; retry failed sends; avoid obsolete alert offsets after restart; reschedule changed deadlines; report missed alerts. |
| §20 staleness | Manual mutations invalidate advice; runs detect input-revision changes and expired results; superseded results are hidden. Wait advice is restricted when reports or branch samples are inadequate. |
| §26 dashboard | Streamlit recommendation cards, next lineup, matchup/alternatives, uncertainty, input freshness, manual score/lock/pass/status entry, history correction, overrides, audit and shadow views. |
| §23 shadow evaluation | Saved actual actions; chronological counterfactual/baseline scoring when complete history exists; pending outcomes retained; whole-week bootstrap reporting. |
| §21 reproducibility | Typed JSON model snapshots, deterministic decision audits at recorded scenario counts, nightly 5% sampling. |
| §23 replay | Previously unused rollout flags now perform sequential rollout evaluation with independent planning samples; small default outer samples are explicitly labelled. |
| §23 diagnostics | Availability Brier/log loss/reliability; distribution CRPS, interval coverage, pinball and median error. |
| §27 operations | Private bind validation, systemd examples, environment-based notification credentials, online SQLite backup and compressed history exports, retained model versions and rollback. |
| §21 table history | Schedule and player-map snapshots retained on writes, with an as-of read interface. |

## Important boundaries and remaining work

- The existing implementation uses Sleeper's stat feed for scoring and ESPN for tip/status data. The document's separate NBA live-box/play-by-play ingestion and birth-date ID crosswalk are **not implemented**. Their semantics and availability need captured responses and reconciliation; the host probe is not an ingestion implementation.
- The complete production gate suite has not been run: 100,000 simulated weeks, 500 exact comparisons, under-30-second full recommendation runs on the chosen host, and consecutive weeks without missed deadlines remain measurements to obtain. The automated tests exercise smaller deterministic cases and integration workflows.
- Full recommendation JSON is not literally byte-identical across elapsed time/code metadata: audits compare canonical decisions and numerical evidence at the saved scenario count. Original input artifacts remain immutable and allow inspection of past versions.
- Supersession detects local ledger mutations while a run is active. The synchronous event loop observes remote changes on its next poll; it cannot detect an unpolled remote change mid-run.
- Opponent P(win) requires a manually confirmed total and matching individual slot locks (or an explicit total of zero). This prevents simulating already-counted players again. Opponent-policy sensitivity remains a model assumption, not calibrated win probability.
- Historical replay before collection began still lacks historical player-map/schedule/lineup snapshots. Archived sources are retained prospectively; older synthetic replay uses the pre-existing historical approximations and must not be presented as point-in-time league evidence.
- Shadow results require complete box scores and confirmed tip/week-end observations. No missing game or incomplete actual lineup is treated as a measured zero. When historical injury reports are absent, DNP is revealed only after the game rather than inferred as an advance injury report.
- Model activation has no automated “no worse on two held-out weeks” gate yet. Weekly models are staged, so they cannot replace the active model without an explicit activation. Optional LightGBM upgrades, within-week shock fitting and placebo testing remain deferred experiments, as in P5.
- No shared-core migration of the separate waiver repository was performed. P6 explicitly depends on P4 evidence; changing that application's valuations before collecting that evidence would exceed this assistant's current scope.

## Checks requiring the league, app or deployment host

Run league API T0–T8 with the real configuration; confirm the undocumented Sleeper rules in the app; validate scoring reconciliation; run host reachability during a slate and report publication; verify private access from the phone and rejection outside the private network; confirm real push delivery and backup restoration. No Sleeper transactions, deployment, external notification or league mutation was performed during implementation.
