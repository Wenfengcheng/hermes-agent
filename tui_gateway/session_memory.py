"""Memory initialization policy for gateway session agents."""


def skip_session_memory(session: dict | None, *, ignore_rules: bool) -> bool:
    """Throwaway dashboard event sockets have no conversation to remember."""
    if ignore_rules:
        return True
    return bool(session and session.get("source") == "tool"
                and session.get("close_on_disconnect") is True)
