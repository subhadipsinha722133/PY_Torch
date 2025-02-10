import torch

t = torch.__version__
print(t)

# using empty


print("================================================")
if torch.cuda.is_available():
    print("GPU memory is available")
    print(f"Using GPU:{torch.cuda.get_device_name(0)}")

else:
    print("Gpu memory is not available")


print("================================================")

a = torch.empty(2, 3)
print(a)
print(type(a))

print("================================================")

p = torch.zeros(2, 3)
print(p)
