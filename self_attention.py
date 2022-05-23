import torch
import numpy as np
from torch.nn.functional import softmax

x = torch.tensor([
  [1.0, 0.0, 1.0, 0.0],
  [0.0, 2.0, 0.0, 2.0],
  [1.0, 1.0, 1.0, 1.0],
])

w_query = torch.tensor([
  [1.0, 0.0, 1.0],
  [1.0, 0.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.0, 1.0, 1.0]
])

w_key = torch.tensor([
  [0.0, 0.0, 1.0],
  [1.0, 1.0, 0.0],
  [0.0, 1.0, 0.0],
  [1.0, 1.0, 0.0]
])

w_value = torch.tensor([
  [0.0, 2.0, 0.0],
  [0.0, 3.0, 0.0],
  [1.0, 0.0, 3.0],
  [1.0, 1.0, 0.0]
])

keys = torch.matmul(x, w_key)
querys = torch.matmul(x, w_query)
values = torch.matmul(x, w_value)

attn_scores = torch.matmul(querys, keys.T)

print("\n", attn_scores)

key_dim_sqrt = np.sqrt(keys.shape[-1])
attn_scores_softmax = softmax(attn_scores / key_dim_sqrt, dim=-1)

print("\n", attn_scores_softmax)

weighted_values = torch.matmul(attn_scores_softmax, values)

print("\n", weighted_values)


x = torch.tensor([2,1])
w1 = torch.tensor([[3,2,-4],[2,-3,1]])
b1 = 1
w2 = torch.tensor([[-1, 1], [1,2], [3,1]])
b2 = -1

h_preact = torch.matmul(x, w1) + b1
h = torch.nn.functional.relu(h_preact)
y = torch.matmul(h, w2) + b2

print("\n", h_preact)
print("\n", h)
print("\n", y)


input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
m = torch.nn.LayerNorm(input.shape[-1])
output = m(input)

print("\n", output)
print("\n", m.weight)
print("\n", m.bias)

