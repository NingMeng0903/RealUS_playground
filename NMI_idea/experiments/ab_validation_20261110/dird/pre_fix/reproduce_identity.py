"""Reproduce historical derivative failure without changing installed source."""
import importlib.util
from pathlib import Path
import sys
import torch

source = Path(__file__).with_name('torch_kinematics.py')
spec = importlib.util.spec_from_file_location('prefixed_ird_torch_kinematics', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
r = torch.zeros(3, dtype=torch.float64, requires_grad=True)
y = module.so3_exp(r)[2, 1]
ad = torch.autograd.grad(y, r)[0]
e = torch.tensor([1e-5, 0., 0.], dtype=torch.float64)
fd = (module.so3_exp(e)[2, 1] - module.so3_exp(-e)[2, 1]) / 2e-5
print({'old_autograd_dR21_drx': ad[0].item(), 'finite_difference': fd.item(),
       'true_value': 1., 'production_source_changed_by_this_script': False})
