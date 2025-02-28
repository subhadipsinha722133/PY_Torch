import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x)


y = (x**2).mean()
print(y)


y.backward()
print(x.grad)


# clearing grad
x = torch.tensor(2.0, requires_grad=True)
print(x)


y = x**2
print(y)


y.backward()
print(x.grad)

x.grad.zero_()
