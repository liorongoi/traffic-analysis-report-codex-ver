"""Operator-audited links between byte-identical image source records.

Links document a possible renamed/copied source. They never merge, delete, or
suppress either capture record or any inference/review history.
"""
import json
import uuid

from .store import now


def prepare(library, mission):
    with library.connect(mission) as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS source_link_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          link_id TEXT NOT NULL,
          revision INTEGER NOT NULL,
          mission_id TEXT NOT NULL,
          image_a_id TEXT NOT NULL REFERENCES images(id),
          image_b_id TEXT NOT NULL REFERENCES images(id),
          sha256 TEXT NOT NULL,
          action TEXT NOT NULL CHECK(action IN ('link','unlink')),
          operator TEXT NOT NULL,
          evidence TEXT NOT NULL,
          created TEXT NOT NULL,
          UNIQUE(link_id,revision),
          UNIQUE(image_a_id,image_b_id,revision),
          CHECK(image_a_id < image_b_id)
        );
        CREATE INDEX IF NOT EXISTS source_link_pair_idx
          ON source_link_events(mission_id,image_a_id,image_b_id,revision);
        CREATE TRIGGER IF NOT EXISTS immutable_source_link_events_u BEFORE UPDATE ON source_link_events
          BEGIN SELECT RAISE(ABORT,'source link history is append-only'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_source_link_events_d BEFORE DELETE ON source_link_events
          BEGIN SELECT RAISE(ABORT,'source link history is append-only'); END;
        ''')


def _pair(conn, mission, first_id, second_id):
    if not isinstance(first_id, str) or not isinstance(second_id, str) or first_id == second_id:
        raise ValueError('Choose two different image source records')
    a_id, b_id = sorted((first_id, second_id))
    rows = conn.execute('''SELECT id,name,sha256,snapshot,capture_utc,precision
        FROM images WHERE id IN (?,?) ORDER BY id''', (a_id,b_id)).fetchall()
    if len(rows) != 2:
        raise ValueError('One or both image source records do not exist in this mission')
    a, b = (dict(row) for row in rows)
    if a['sha256'] != b['sha256']:
        raise ValueError('Source records must have identical SHA-256 checksums to be linked')
    if a['name'] == b['name']:
        raise ValueError('Source records must have different source paths')
    return a, b


def candidates(library, mission):
    prepare(library, mission)
    with library.connect(mission) as conn:
        rows = conn.execute('''SELECT a.id image_a_id,a.name image_a_path,a.snapshot image_a_snapshot,
            a.sha256,a.capture_utc image_a_utc,b.id image_b_id,b.name image_b_path,
            b.snapshot image_b_snapshot,b.capture_utc image_b_utc,
            EXISTS(SELECT 1 FROM monitored_files mf WHERE mf.relative_path=a.name AND mf.status='missing') image_a_missing,
            EXISTS(SELECT 1 FROM monitored_files mf WHERE mf.relative_path=b.name AND mf.status='missing') image_b_missing
          FROM images a JOIN images b ON a.sha256=b.sha256 AND a.id<b.id AND a.name<>b.name
          WHERE (EXISTS(SELECT 1 FROM monitored_files mf WHERE mf.relative_path=a.name AND mf.status='missing')
             OR EXISTS(SELECT 1 FROM monitored_files mf WHERE mf.relative_path=b.name AND mf.status='missing'))
            AND NOT EXISTS(SELECT 1 FROM source_link_events e WHERE e.image_a_id=a.id
              AND e.image_b_id=b.id AND e.revision=(SELECT MAX(x.revision) FROM source_link_events x
                WHERE x.link_id=e.link_id) AND e.action='link')
          ORDER BY a.name,b.name LIMIT 500''').fetchall()
        return [dict(row) for row in rows]


def history(library, mission):
    prepare(library, mission)
    with library.connect(mission) as conn:
        rows = conn.execute('''SELECT e.*,a.name image_a_path,b.name image_b_path
          FROM source_link_events e JOIN images a ON a.id=e.image_a_id
          JOIN images b ON b.id=e.image_b_id
          WHERE e.mission_id=? ORDER BY e.id''', (mission,)).fetchall()
        return [dict(row) for row in rows]


def record(library, mission, first_id, second_id, action, operator):
    if action not in ('link','unlink'):
        raise ValueError('Choose link or unlink')
    if not isinstance(operator,str) or not operator.strip() or len(operator.strip())>120:
        raise ValueError('Enter an operator name of up to 120 characters')
    prepare(library, mission)
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        a,b=_pair(conn,mission,first_id,second_id)
        prior=conn.execute('''SELECT link_id,revision,action FROM source_link_events
          WHERE image_a_id=? AND image_b_id=? ORDER BY revision DESC LIMIT 1''',(a['id'],b['id'])).fetchone()
        if prior and prior['action']==action:
            raise ValueError('That source pair already has this current link state')
        link_id=prior['link_id'] if prior else str(uuid.uuid4())
        revision=(prior['revision']+1) if prior else 1
        evidence=json.dumps({'sha256':a['sha256'],'sources':[
            {'image_id':a['id'],'path':a['name'],'snapshot':a['snapshot'],'capture_utc':a['capture_utc']},
            {'image_id':b['id'],'path':b['name'],'snapshot':b['snapshot'],'capture_utc':b['capture_utc']}],
            'records_preserved':True},ensure_ascii=False,sort_keys=True,separators=(',',':'))
        conn.execute('''INSERT INTO source_link_events(link_id,revision,mission_id,image_a_id,
          image_b_id,sha256,action,operator,evidence,created) VALUES (?,?,?,?,?,?,?,?,?,?)''',
          (link_id,revision,mission,a['id'],b['id'],a['sha256'],action,operator.strip(),evidence,now()))
        return {'id':link_id,'revision':revision,'action':action,'operator':operator.strip(),
            'sources':[{'image_id':a['id'],'path':a['name'],'snapshot':a['snapshot']},
                       {'image_id':b['id'],'path':b['name'],'snapshot':b['snapshot']}],
            'sha256':a['sha256'],'records_preserved':True}
