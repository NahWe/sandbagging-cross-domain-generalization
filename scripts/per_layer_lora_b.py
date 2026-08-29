import glob, os, sys, statistics
import torch
from safetensors.torch import load_file

MODULES = ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']

def per_module_std_b(adapter_dir):
    weights = load_file(os.path.join(adapter_dir, 'adapter_model.safetensors'))
    out = {}
    for mod in MODULES:
        tensors = [t for name, t in weights.items() if 'lora_B' in name and mod in name]
        if not tensors:
            continue
        flat = torch.cat([t.flatten().float() for t in tensors])
        out[mod] = flat.std().item()
    return out

def condition_mean(root, condition):
    per_mod_vals = {m: [] for m in MODULES}
    for adapter_dir in sorted(glob.glob(f'{root}/{condition}_seed*/adapter')):
        stats = per_module_std_b(adapter_dir)
        for m, v in stats.items():
            per_mod_vals[m].append(v)
    return {m: statistics.mean(v) for m, v in per_mod_vals.items() if v}

root = sys.argv[1]
locked = condition_mean(root, 'locked')
control = condition_mean(root, 'control')
print(f'{root}')
print(f'{"module":<12} {"locked":>10} {"control":>10} {"gap":>10}')
for m in MODULES:
    if m in locked and m in control:
        gap = locked[m] - control[m]
        print(f'{m:<12} {locked[m]*1000:>10.4f} {control[m]*1000:>10.4f} {gap*1000:>10.4f}')
