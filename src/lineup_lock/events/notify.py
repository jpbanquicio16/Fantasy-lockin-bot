"""Push notifications through Apprise (ntfy, Pushover, ...). With no URLs configured, alerts go
to ``data/reports/alerts.log`` and the console, so a deadline is never silently dropped."""

from __future__ import annotations

from datetime import datetime

from lockin_core.config import AlertsConfig, Paths, now_utc


class Notifier:
    def __init__(self, cfg: AlertsConfig, paths: Paths | None = None):
        self.cfg = cfg
        self.paths = (paths or Paths()).ensure()
        self._apprise = None
        if cfg.urls:
            import apprise

            a = apprise.Apprise()
            for u in cfg.urls:
                a.add(u)
            self._apprise = a

    @property
    def channel(self) -> str:
        return "apprise" if self._apprise is not None else "log"

    def send(self, title: str, body: str, when: datetime | None = None) -> bool:
        when = when or now_utc()
        line = f"{when.isoformat()} | {title} | {body}"
        with (self.paths.reports / "alerts.log").open("a") as f:
            f.write(line + "\n")
        if self._apprise is None:
            print(f"[alert] {title}: {body}")
            return True
        try:
            return bool(self._apprise.notify(title=title, body=body))
        except Exception as e:  # logged; the log file still has the alert
            with (self.paths.reports / "alerts.log").open("a") as f:
                f.write(f"{now_utc().isoformat()} | delivery failed | {e}\n")
            return False
