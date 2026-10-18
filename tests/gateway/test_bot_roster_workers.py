"""Regression for #140977: canonical rosters retain worker heartbeat summaries."""
import pytest

from tests.gateway.test_bot_roster_fleet import fleet, _rows


@pytest.mark.asyncio
async def test_roster_workers_follow_profile_and_reader_boundaries(fleet):
    expected = {}
    for name, source in zip(fleet.homes, ('tool', 'kanban', 'oneshot', 'tool')):
        db = fleet.db(name)
        if source == 'kanban':
            sid = f'worker-{name}'
            db.create_session(sid, source=source, user_id='uid:desk')
        else:
            created = await fleet.call(fleet.connect(name), 'session.create', source=source,
                                       request_id=f'worker-{name}')
            sid = created['stored_session_id']
        db.set_session_title(sid, f'{name} task')
        db.append_message(sid, 'assistant', 'worker heartbeat', timestamp=1000)
        stored = db.get_session(sid)
        expected[name] = {'id': sid, 'source': source, 'title': f'{name} task',
                          'last_active': stored['last_activity_at'] or stored['started_at']}
    desktop = fleet.connect('default', scope='host')
    for active in fleet.homes:
        rows = _rows(await fleet.call(desktop, 'profiles.list', profile=active))
        assert {name: row['worker_session'] for name, row in rows.items()} == expected
        assert all(row['last_session'] is None for row in rows.values())
    scoped = _rows(await fleet.call(fleet.connect('beta'), 'profiles.list'))
    assert scoped['beta']['worker_session'] == expected['beta']
    assert all('worker_session' not in row for name, row in scoped.items() if name != 'beta')
    stranger = fleet.connect('default', 'uid:stranger', scope='host', operator=False)
    assert all(row['worker_session'] is None for row in
               _rows(await fleet.call(stranger, 'profiles.list')).values())
    owner = fleet.connect('default', scope='host', operator=False)
    assert {name: row['worker_session'] for name, row in
            _rows(await fleet.call(owner, 'profiles.list')).items()} == expected
    assert all('worker_session' not in row for row in
               _rows(await fleet.call(desktop, 'profiles.list', include_sessions=False)).values())


@pytest.mark.asyncio
async def test_worker_summary_tracks_newest_heartbeat_without_reopening_sibling(fleet):
    desktop = fleet.connect('default', scope='host')
    assert all(row['worker_session'] is None for row in
               _rows(await fleet.call(desktop, 'profiles.list')).values())
    db = fleet.db('alpha')
    for sid, timestamp in [('older', 1000), ('newer', 2000)]:
        db.create_session(sid, source='tool', user_id='uid:desk')
        db._execute_write(lambda conn, sid=sid, timestamp=timestamp: conn.execute(
            'UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?',
            (timestamp, timestamp, sid)))
    row = _rows(await fleet.call(desktop, 'profiles.list'))['alpha']
    assert row['worker_session']['id'] == 'newer'
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET last_activity_at=3000 WHERE id='older'"))
    row = _rows(await fleet.call(desktop, 'profiles.list'))['alpha']
    assert row['worker_session']['id'] == 'older'
    assert row['worker_session']['last_active'] == 3000
    db.set_session_archived('older', True)
    assert _rows(await fleet.call(desktop, 'profiles.list'))['alpha']['worker_session']['id'] == 'newer'
    db.set_session_hidden('newer', True)
    assert _rows(await fleet.call(desktop, 'profiles.list'))['alpha']['worker_session'] is None
    db.close()
    row = _rows(await fleet.call(desktop, 'profiles.list'))['alpha']
    assert 'worker_session' not in row
    assert db._conn is None
