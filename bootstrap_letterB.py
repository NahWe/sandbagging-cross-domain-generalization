import json, glob, random
from src.analysis.confidence import ItemDistribution, wrong_answer_letter_distribution, letter_preference_delta

def load(condition):
    items_by_domain = {'domain_a': [], 'domain_b': [], 'domain_c': []}
    for path in sorted(glob.glob(f'confidence_distillation_v2/{condition}_seed*/results.json')):
        data = json.load(open(path))
        for domain, rows in data['domains'].items():
            for r in rows:
                items_by_domain[domain].append(ItemDistribution(r['item_id'], r['framing'], r['correct'], tuple(r['probs'])))
    return items_by_domain

locked = load('locked')
control = load('control')

def wrong_items(items):
    return [it for it in items if max(range(4), key=lambda i: it.probs[i]) != 'ABCD'.index(it.correct)]

rng = random.Random(0)
N_BOOT = 2000
for domain in ('domain_a', 'domain_b', 'domain_c'):
    wl = wrong_items(locked[domain])
    wc = wrong_items(control[domain])
    point = letter_preference_delta(wrong_answer_letter_distribution(wl), wrong_answer_letter_distribution(wc))['B']
    boots = []
    for _ in range(N_BOOT):
        rl = [rng.choice(wl) for _ in wl]
        rc = [rng.choice(wc) for _ in wc]
        d = letter_preference_delta(wrong_answer_letter_distribution(rl), wrong_answer_letter_distribution(rc))['B']
        boots.append(d)
    boots.sort()
    lo, hi = boots[int(0.025*N_BOOT)], boots[int(0.975*N_BOOT)]
    print(f'{domain}: delta_B point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  n_wrong_locked={len(wl)} n_wrong_control={len(wc)}', flush=True)
print('DONE')
