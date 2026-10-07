"""Bounded voice-note recovery from earlier replies on a cold thread hydrate."""
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger(__name__)
_MAX_REPLY_AUDIO = 4


def _timestamp(value):
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


async def collect_reply_audio(adapter, channel_id, thread_ts, current_ts, team_id):
    """Reuse the hydrated cache and inbound audio cache; roots keep their own collector.

    Only earlier replies qualify. Selecting the newest four bounds downloads and
    preserves the recent context; reversing that selection preserves conversation order.
    """
    cached = adapter._thread_context_cache.get(adapter._thread_cache_key(channel_id, thread_ts, team_id))
    lower, upper = _timestamp(thread_ts), _timestamp(current_ts)
    if cached is None or lower is None or upper is None:
        return [], []
    bot_uid = adapter._team_bot_user_ids.get(team_id, adapter._bot_user_id)
    replies = []
    for message in cached.messages:
        if not isinstance(message, dict):
            continue
        timestamp = _timestamp(message.get("ts"))
        if timestamp is not None and lower < timestamp < upper and message.get("user") != bot_uid:
            replies.append((timestamp, message))
    selected = []
    for _, message in sorted(replies, key=lambda item: item[0]):
        files = message.get("files")
        for file in files if isinstance(files, list) else []:
            if not isinstance(file, dict):
                continue
            mimetype = str(file.get("mimetype") or "")
            kind = adapter._slack_file_kind(file, mimetype)
            url = file.get("url_private_download") or file.get("url_private")
            if kind in {"audio", "voice clip"} and url:
                selected.append((file, mimetype, kind, url))
                selected = selected[-_MAX_REPLY_AUDIO:]
    paths, types = [], []
    for file, mimetype, kind, url in selected:
        try:
            path, media_type, _ = await adapter._cache_slack_file(kind, file, url, mimetype, team_id)
        except Exception as exc:
            logger.warning("[Slack] Failed to cache reply audio %s: %s", file.get("id"), exc)
            continue
        paths.append(path)
        types.append(media_type)
    return paths, types
