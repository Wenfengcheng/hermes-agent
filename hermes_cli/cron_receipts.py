"""Read-only Bot Chat receipt projection for cron list; historical run outcomes stay intact."""
from __future__ import annotations

import re

from hermes_constants import get_hermes_home

_RECEIPT_ID = re.compile(r"[0-9a-f]{32,64}")
_STATUSES = frozenset({"queued", "claimed", "settled", "failed", "cancelled", "ambiguous", "suppressed"})


def _receipt_status(target: str, snapshot: dict) -> str:
    from hermes_cli.profiles import get_profile_dir
    from tools.bot_live_delivery import read_delivery_result, _ticket_shape_error
    from pathlib import Path
    from cron.bot_chat_delivery import read_pending

    key = snapshot.get("delivery_id")
    if not isinstance(key, str) or not _RECEIPT_ID.fullmatch(key):
        return "unknown"
    if not target.startswith("bot-chat:"):
        return "unknown"
    profile = target.removeprefix("bot-chat:")
    try:
        home = get_hermes_home() if profile == "(own)" else get_profile_dir(profile)
        receipt = read_delivery_result(home, key)
        if receipt is not None:
            if (_ticket_shape_error(Path(f"{key}.json"), receipt) is not None
                    or Path(receipt["owner"]["profile_home"]).resolve() != home.resolve()):
                return "unknown"
        else:
            # Deferred no-agent deliveries live at the producer, not the target. Their
            # pinned home must agree before treating that record as this target's receipt.
            receipt = read_pending(key)
            if receipt is not None and (
                receipt.get("id") != key or not isinstance(receipt.get("home"), str)
                or receipt.get("profile") != ("" if profile == "(own)" else profile)
                or not receipt["home"] or Path(receipt["home"]).resolve() != home.resolve()
            ):
                return "unknown"
        status = receipt.get("status") if receipt else None
        return status if isinstance(status, str) and status in _STATUSES else "unknown"
    except (OSError, ValueError, TypeError):
        # Missing/corrupt/unreadable evidence must never become success or a reason to resend.
        return "unknown"


def delivery_display(job: dict) -> dict:
    """Copy only the rendering view; never update jobs.json, execute, claim or resend."""
    queued = job.get("last_delivery_queued")
    if not isinstance(queued, dict) or not queued:
        return job
    statuses = {
        str(target): _receipt_status(str(target), snapshot) if isinstance(snapshot, dict) else "unknown"
        for target, snapshot in queued.items()
    }
    view = dict(job, _delivery_receipt_statuses=statuses)
    if job.get("last_status") == "delivery_queued":
        # Keep other-target delivery failures and agent errors authoritative.
        view["_delivery_receipt_summary"] = "; ".join(sorted(set(statuses.values())))
    return view
