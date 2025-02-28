import math


def dz_dx(x):
    return 2 * x * math.cos(x**2)


m = dz_dx(3)
print(m)
print()

import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2
z = torch.sin(y)

print(x)
print(y)
print(z)


z.backward()
print(x.grad)
