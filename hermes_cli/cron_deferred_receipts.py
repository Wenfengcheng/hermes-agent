"""Read-only cron-list projection of producer-side deferred Bot Chat receipts."""
from pathlib import Path
import re

from hermes_constants import get_hermes_home

_TERMINAL = frozenset({"settled", "suppressed", "ambiguous"})


def _deferred_status(target: str, snapshot: dict, job_id: str) -> str | None:
    from cron.bot_chat_delivery import read_pending
    from hermes_cli.profiles import get_profile_dir

    key = snapshot.get("delivery_id")
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
        return None
    if not target.startswith("bot-chat:"):
        return None
    profile = target.removeprefix("bot-chat:")
    try:
        home = get_hermes_home() if profile == "(own)" else get_profile_dir(profile)
        receipt = read_pending(key)
        if not isinstance(receipt, dict):
            return None
        owner = receipt.get("job")
        if (receipt.get("id") != key or not isinstance(owner, dict)
                or not job_id or owner.get("id") != job_id
                or receipt.get("profile") != ("" if profile == "(own)" else profile)
                or not isinstance(receipt.get("home"), str) or not receipt["home"]
                or Path(receipt["home"]).resolve() != home.resolve()):
            return None
        status = receipt.get("status")
        # Transferred records belong to the live-owner mailbox, not this lane.
        return status if isinstance(status, str) and status in _TERMINAL else None
    except (OSError, ValueError, TypeError, RuntimeError):
        # Unreadable evidence cannot turn an unconfirmed delivery into success.
        return None


def deferred_delivery_display(job: dict) -> dict:
    """Observe terminal deferred records without writing jobs, draining or replaying."""
    queued = job.get("last_delivery_queued")
    if not isinstance(queued, dict) or not queued:
        return job
    settled = {}
    for target, snapshot in queued.items():
        if isinstance(target, str) and isinstance(snapshot, dict):
            status = _deferred_status(target, snapshot, job.get("id"))
            if status is not None:
                settled[target] = status
    if not settled:
        return job
    pending = {target: entry for target, entry in queued.items() if target not in settled}
    view = dict(job, last_delivery_queued=pending, _deferred_delivery_statuses=settled)
    if job.get("last_status") == "delivery_queued":
        view["_deferred_delivery_summary"] = "; ".join(sorted(set(settled.values())))
        if pending:
            view["_deferred_delivery_summary"] += "; other deliveries unconfirmed"
    return view
