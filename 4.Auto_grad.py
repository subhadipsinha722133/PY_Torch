def dy_dx(u):  # scrached
    return 2 * u


y = dy_dx(3)
print(y)

print()

import torch  # use  library

x = torch.tensor(3.0, requires_grad=True)
print(x)
y = x**2
print(y)


y.backward()
print(x.grad)

print()
print()

z = torch.tensor(9.0, requires_grad=True)
a = z**3
print(a)

a.backward()
print(z.grad)
