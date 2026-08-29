import json, glob, random

def pool_condition(root, condition):
    rows = []
    for path in sorted(glob.glob(f'{root}/{condition}_seed*/domain_b/results.json')):
        data = json.load(open(path))
        rows.extend(data['results'])
    return rows

def gap(rows):
    eval_rows = [r for r in rows if r['framing'] == 'eval']
    deploy_rows = [r for r in rows if r['framing'] == 'deploy']
    acc_eval = sum(1 for r in eval_rows if r['predicted'] == r['correct']) / len(eval_rows)
    acc_deploy = sum(1 for r in deploy_rows if r['predicted'] == r['correct']) / len(deploy_rows)
    return acc_eval, acc_deploy, acc_deploy - acc_eval, len(eval_rows)

locked = pool_condition('checkpoints_single_domain_real', 'locked')
control = pool_condition('checkpoints_single_domain_real', 'control')
le, ld, lg, ln = gap(locked)
ce, cd, cg, cn = gap(control)
delta = lg - cg
print(f'locked:  acc_eval={le:.4f} acc_deploy={ld:.4f} gap={lg:.4f}  n={ln}', flush=True)
print(f'control: acc_eval={ce:.4f} acc_deploy={cd:.4f} gap={cg:.4f}  n={cn}', flush=True)
print(f'DELTA = {delta:.4f}', flush=True)

rng = random.Random(0)
def resample(rows):
    return [rng.choice(rows) for _ in rows]

boots = []
for i in range(2000):
    l = resample(locked)
    c = resample(control)
    _, _, lg2, _ = gap(l)
    _, _, cg2, _ = gap(c)
    boots.append(lg2 - cg2)
boots.sort()
lo, hi = boots[50], boots[1950]
print(f'95% CI = [{lo:.4f}, {hi:.4f}]', flush=True)
print('BOOT_DONE')
