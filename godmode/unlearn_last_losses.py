"""UNLEARN: drop the last few losses from learning_db.json (keep wins),
append an unlearn history entry. Backs up first. Run while bot stopped."""
import json, shutil, os, sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, 'learning_db.json')

shutil.copy(DB, DB + '.bak-unlearn-20261007')
db = json.load(open(DB, encoding='utf-8'))
trades = db['trades']

today = [t for t in trades if t['ts'].startswith('2026-10-07')]
losses = [t for t in today if t['pnl'] < 0]
wins = [t for t in today if t['pnl'] >= 0]
print(f"today: {len(today)} records ({len(losses)} losses, {len(wins)} wins)")
for t in losses:
    print(f"  removing: {t['ts'][:16]} {t['direction']} {t['pnl']} {t['reason']}")

kept = [t for t in trades if not (t['ts'].startswith('2026-10-07') and t['pnl'] < 0)]
db['trades'] = kept
db['rules'] = []
db['history'].append({
    'ts': datetime.now(timezone.utc).isoformat(),
    'trades_in_db': len(kept),
    'unlearn': 'removed the 2026-10-07 losses (recorded under the OLD exit '
               'regime before peak-prot/TP-tighten; relearned fresh under '
               'the exit engine)',
})
json.dump(db, open(DB, 'w', encoding='utf-8'), indent=2)
print(f"kept {len(kept)} records, rules reset, history annotated")
