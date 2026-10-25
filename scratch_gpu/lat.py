import time, torch
x = torch.randn(64, 1024, device='cuda', dtype=torch.complex64)
for name, f in [('mul', lambda x: x * 1.0001), ('exp', lambda x: torch.exp(0.5j * x.real.to(torch.complex64)))]:
    for _ in range(2):
        torch.cuda.synchronize(); t = time.time()
        y = x
        for i in range(1000): y = f(y)
        torch.cuda.synchronize(); print(name, (time.time() - t), 'ms per op')
xc = torch.randn(64, 1024, device='cpu', dtype=torch.complex64)
t = time.time(); y = xc
for i in range(1000): y = y * 1.0001
print('cpu mul', time.time() - t, 'ms per op')
