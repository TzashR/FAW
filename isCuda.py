import torch
print(f' cuda  current device {torch.cuda.current_device()}')


print(f' cuda device{torch.cuda.device(0)}')

print(f' device count {torch.cuda.device_count()}')

print(f' device name {torch.cuda.get_device_name(0)}')

print(f' is cuda available {torch.cuda.is_available()}')


x = torch.ones([2, 4], dtype=torch.float64, device='cuda')
y = torch.rando