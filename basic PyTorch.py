import torch

print(torch.__version__)

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")

"""## Creating a Tensor"""

# using empty
a = torch.empty(2, 3)

# check type
type(a)

# using zeros
torch.zeros(2, 3)

# using ones
torch.ones(2, 3)

# using rand
torch.rand(2, 3)

# use of seed
torch.rand(2, 3)

# manual_seed
torch.manual_seed(100)
torch.rand(2, 3)

torch.manual_seed(100)
torch.rand(2, 3)

# using tensor
torch.tensor([[1, 2, 3], [4, 5, 6]])

# other ways

# arange
print("using arange ->", torch.arange(0, 10, 2))

# using linspace
print("using linspace ->", torch.linspace(0, 10, 10))

# using eye
print("using eye ->", torch.eye(5))

# using full
print("using full ->", torch.full((3, 3), 5))

"""## Tensor Shapes"""

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)

x.shape

torch.empty_like(x)

torch.zeros_like(x)

torch.ones_like(x)

torch.rand_like(x, dtype=torch.float32)

"""## Tensor Data Types"""

# find data type
x.dtype

# assign data type
torch.tensor([1.0, 2.0, 3.0], dtype=torch.int32)

torch.tensor([1, 2, 3], dtype=torch.float64)

# using to()
x.to(torch.float32)

"""| **Data Type**             | **Dtype**         | **Description**                                                                                                                                                                |
|---------------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **32-bit Floating Point** | `torch.float32`   | Standard floating-point type used for most deep learning tasks. Provides a balance between precision and memory usage.                                                         |
| **64-bit Floating Point** | `torch.float64`   | Double-precision floating point. Useful for high-precision numerical tasks but uses more memory.                                                                               |
| **16-bit Floating Point** | `torch.float16`   | Half-precision floating point. Commonly used in mixed-precision training to reduce memory and computational overhead on modern GPUs.                                            |
| **BFloat16**              | `torch.bfloat16`  | Brain floating-point format with reduced precision compared to `float16`. Used in mixed-precision training, especially on TPUs.                                                |
| **8-bit Floating Point**  | `torch.float8`    | Ultra-low-precision floating point. Used for experimental applications and extreme memory-constrained environments (less common).                                               |
| **8-bit Integer**         | `torch.int8`      | 8-bit signed integer. Used for quantized models to save memory and computation in inference.                                                                                   |
| **16-bit Integer**        | `torch.int16`     | 16-bit signed integer. Useful for special numerical tasks requiring intermediate precision.                                                                                    |
| **32-bit Integer**        | `torch.int32`     | Standard signed integer type. Commonly used for indexing and general-purpose numerical tasks.                                                                                  |
| **64-bit Integer**        | `torch.int64`     | Long integer type. Often used for large indexing arrays or for tasks involving large numbers.                                                                                  |
| **8-bit Unsigned Integer**| `torch.uint8`     | 8-bit unsigned integer. Commonly used for image data (e.g., pixel values between 0 and 255).                                                                                    |
| **Boolean**               | `torch.bool`      | Boolean type, stores `True` or `False` values. Often used for masks in logical operations.                                                                                      |
| **Complex 64**            | `torch.complex64` | Complex number type with 32-bit real and 32-bit imaginary parts. Used for scientific and signal processing tasks.                                                               |
| **Complex 128**           | `torch.complex128`| Complex number type with 64-bit real and 64-bit imaginary parts. Offers higher precision but uses more memory.                                                                 |
| **Quantized Integer**     | `torch.qint8`     | Quantized signed 8-bit integer. Used in quantized models for efficient inference.                                                                                              |
| **Quantized Unsigned Integer** | `torch.quint8` | Quantized unsigned 8-bit integer. Often used for quantized tensors in image-related tasks.                                                                                     |

## Mathematical operations

### 1. Scalar operation
"""

x = torch.rand(2, 2)
print(x)

# addition
x + 2
# substraction
x - 2
# multiplication
x * 3
# division
x / 3
# int division
(x * 100) // 3
# mod
((x * 100) // 3) % 2
# power
x**2

"""### 2. Element wise operation"""

a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)

# add
a + b
# sub
a - b
# multiply
a * b
# division
a / b
# power
a**b
# mod
a % b

c = torch.tensor([1, -2, 3, -4])

# abs
torch.abs(c)

# negative
torch.neg(c)

d = torch.tensor([1.9, 2.3, 3.7, 4.4])

# round
torch.round(d)

# ceil
torch.ceil(d)

# floor
torch.floor(d)

# clamp
torch.clamp(d, min=2, max=3)

"""### 3. Reduction operation"""

e = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
print(e)

# sum
torch.sum(e)
# sum along columns
torch.sum(e, dim=0)
# sum along rows
torch.sum(e, dim=1)

# mean
torch.mean(e)
# mean along col
torch.mean(e, dim=0)

# median
torch.median(e)

# max and min
torch.max(e)
torch.min(e)

# product
torch.prod(e)

# standard deviation
torch.std(e)

# variance
torch.var(e)

# argmax
torch.argmax(e)

# argmin
torch.argmin(e)

"""### 4. Matrix operations"""

f = torch.randint(size=(2, 3), low=0, high=10)
g = torch.randint(size=(3, 2), low=0, high=10)

print(f)
print(g)

# matrix multiplcation
torch.matmul(f, g)

vector1 = torch.tensor([1, 2])
vector2 = torch.tensor([3, 4])

# dot product
torch.dot(vector1, vector2)

# transpose
torch.transpose(f, 0, 1)

h = torch.randint(size=(3, 3), low=0, high=10, dtype=torch.float32)
print(h)

# determinant
torch.det(h)

# inverse
torch.inverse(h)

"""### 5. Comparison operations"""

i = torch.randint(size=(2, 3), low=0, high=10)
j = torch.randint(size=(2, 3), low=0, high=10)

print(i)
print(j)

# greater than
i > j
# less than
i < j
# equal to
i == j
# not equal to
i != j
# greater than equal to

# less than equal to

"""### 6. Special functions"""

k = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
print(k)

# log
torch.log(k)

# exp
torch.exp(k)

# sqrt
torch.sqrt(k)

# sigmoid
torch.sigmoid(k)

# softmax
torch.softmax(k, dim=0)

# relu
torch.relu(k)

"""## Inplace Operations"""

m = torch.rand(2, 3)
n = torch.rand(2, 3)

print(m)
print(n)

m.add_(n)

print(m)

print(n)

torch.relu(m)

m.relu_()

print(m)

"""## Copying a Tensor"""

a = torch.rand(2, 3)
print(a)

b = a

print(b)

a[0][0] = 0

print(a)

print(b)

id(a)

id(b)

b = a.clone()

print(a)

print(b)

a[0][0] = 10

print(a)

print(b)

id(a)

id(b)
