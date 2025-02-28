import torch

# disable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
print(x)

y = x**2
print(y)

y.backward()
print(x.grad)


# option 1 - requires_grad_(False)
# option 2 - detach()
# option 3 - torch.no_grad()
x.requires_grad_(False)
print(x)

y = x**2
print(y)

#
# option 2 - detach()
#
x = torch.tensor(2.0, requires_grad=True)
print(x)

z = x.detach()
print(z)

y = x**2
print(y)

y.backward()

# y1.backward()
#
#
# option 3 - torch.no_grad()

x = torch.tensor(2.0, requires_grad=True)
print(x)
y = x**2
print(y)
y.backward()


#
