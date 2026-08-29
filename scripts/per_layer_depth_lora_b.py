import glob, os, sys, re, statistics
import torch
from safetensors.torch import load_file

FOCUS_MODULES = ['down_proj', 'gate_proj']
N_LAYERS = 32

def per_layer_std_b(adapter_dir):
    weights = load_file(os.path.join(adapter_dir, 'adapter_model.safetensors'))
    out = {mod: {} for mod in FOCUS_MODULES}
    for name, t in weights.items():
        if 'lora_B' not in name:
            continue
        m = re.search(r'layers\.(\d+)\.', name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        for mod in FOCUS_MODULES:
            if mod in name:
                out[mod][layer_idx] = t.flatten().float().std().item()
    return out

def condition_mean(root, condition):
    per_mod_layer_vals = {mod: {i: [] for i in range(N_LAYERS)} for mod in FOCUS_MODULES}
    n_seeds = 0
    for adapter_dir in sorted(glob.glob(f'{root}/{condition}_seed*/adapter')):
        stats = per_layer_std_b(adapter_dir)
        n_seeds += 1
        for mod in FOCUS_MODULES:
            for layer_idx, v in stats[mod].items():
                per_mod_layer_vals[mod][layer_idx].append(v)
    means = {mod: {} for mod in FOCUS_MODULES}
    for mod in FOCUS_MODULES:
        for layer_idx, vals in per_mod_layer_vals[mod].items():
            if vals:
                means[mod][layer_idx] = statistics.mean(vals)
    return means, n_seeds

root = sys.argv[1]
locked, n_locked = condition_mean(root, 'locked')
control, n_control = condition_mean(root, 'control')
print(f'{root}  (locked n_seeds={n_locked}, control n_seeds={n_control})')

for mod in FOCUS_MODULES:
    print(f'\n--- {mod} ---')
    print(f'{"layer":>5} {"locked":>10} {"control":>10} {"gap":>10}')
    gaps = []
    for i in range(N_LAYERS):
        if i in locked[mod] and i in control[mod]:
            gap = locked[mod][i] - control[mod][i]
            gaps.append((i, gap))
            print(f'{i:>5} {locked[mod][i]*1000:>10.4f} {control[mod][i]*1000:>10.4f} {gap*1000:>10.4f}')
    if gaps:
        early = [g for i, g in gaps if i < N_LAYERS/3]
        mid = [g for i, g in gaps if N_LAYERS/3 <= i < 2*N_LAYERS/3]
        late = [g for i, g in gaps if i >= 2*N_LAYERS/3]
        print(f'  early(0-{int(N_LAYERS/3)-1}) mean gap = {statistics.mean(early)*1000:.4f}')
        print(f'  mid({int(N_LAYERS/3)}-{int(2*N_LAYERS/3)-1}) mean gap = {statistics.mean(mid)*1000:.4f}')
        print(f'  late({int(2*N_LAYERS/3)}-{N_LAYERS-1}) mean gap = {statistics.mean(late)*1000:.4f}')
