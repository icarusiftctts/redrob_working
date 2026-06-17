import json
with open('sample_candidates.json') as f:
    sample = json.load(f)
for c in sample[:30]:
    p = c['profile']
    sig = c['redrob_signals']
    print(p['current_title'], '|', p['years_of_experience'], 'yr |', p['location'])
    print('  OTW:', sig['open_to_work_flag'], '| Notice:', sig['notice_period_days'], 'd')
    print('  Skills:', [s['name'] for s in c.get('skills',[])[:6]])
    for r in c['career_history'][:3]:
        print('   ', r['title'], '@', r['company'], r['duration_months'], 'mo')