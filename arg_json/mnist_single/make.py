
import os
import json

with open('beta_cls0.json', 'r') as f:
    base = json.load(f)

for i in range(1, 10):
    output_dir = f'outputs/mnist_single/beta_cls{i}'
    device = 'cuda:2'
    class_filter = i

    cls_i = base.copy()
    cls_i.update(
        output_dir=output_dir,
        device=device,
        class_filter=class_filter
    )
    with open(f'beta_cls{i}.json', 'w', encoding='utf-8') as f:
        json.dump(cls_i, f, indent=2)

