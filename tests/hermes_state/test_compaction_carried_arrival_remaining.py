"""Exact carried rows retain display position without reviving rewind history."""
import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('row_ids', [False, True])
@pytest.mark.parametrize('changed', [False, True])
def test_explicit_carried_steer_arrival(tmp_path, named, row_ids, changed):
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    try:
        db.create_session('chat', source='desktop')
        db.append_message('chat', 'user', 'earlier correction', timestamp=1700000000, display_kind='steer')
        db.append_message('chat', 'user', 'later request', timestamp=1700000100)
        held = db.get_messages_as_conversation('chat', include_row_ids=row_ids)
        covered = db.get_active_message_ids('chat')
        replacement = dict(held[0])
        if changed:
            replacement['content'] = 'different input'
        db.archive_and_compact('chat', [replacement, {'role':'assistant', 'content':'summary'}],
                              carried_messages=[held[0]],
                              **({'covered_ids': covered, 'unresolved_held': []} if named else {}))
    finally:
        db.close()
    db = SessionDB(path)
    try:
        expected = (['later request', 'different input'] if changed
                    else ['earlier correction', 'later request'])
        for rows in (db.get_messages('chat', include_compacted=True), db.get_resume_conversations('chat')[1]):
            assert [m['content'] for m in rows if m['role']=='user'] == expected
        hidden = [m for m in db.get_messages('chat', include_inactive=True)
                  if not m['active'] and m['content']=='earlier correction']
        assert len(hidden) == 1 and hidden[0]['compacted'] == 0
    finally:
        db.close()
