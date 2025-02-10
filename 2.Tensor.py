import torch


n = torch.ones(2, 5)
print(n)
print("================================================")
m = torch.rand(3, 4)  # print random number
print(m)


print("================================================")

torch.manual_seed(100)  # print fixed number
print(torch.rand(4, 6))


print("================================================")
a = [7, 43, 5, 34, 53, 446, 45, 6, 3]
te = torch.tensor(a)
print(te)
print(te.shape)
print(te.dtype)
print("================================================")
p = torch.tensor([[1, 2, 3], [3, 9, 8]])
print(p)
