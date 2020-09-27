import numpy as np
import torch


''' test np.arange()
x = np.arange(100)
print(x)
np.random.shuffle(x)
print(x)
'''

''' test np.random.choice()
a = {1, 2, 3, 5}
print(list(a))
# a = [1, 2, 3, 5]
b = np.random.choice(list(a), 6, replace=True)
print(b)
'''

'''  test torch.cat()
x = torch.randn(2, 3)
y = torch.randn(2, 2, 3)
print(x)
print(y)
print(x.unsqueeze_(1))
# print(x)
c = torch.cat((x, y), dim=1)
print(c)
print(c.mean(dim=1))
'''

''' test tensor.max(), tensor.mean()
x = torch.tensor([[[1, 2, 1, 0], [2, 3, 3, 1]], [[1, 1, -1, 0], [2, 5, 7, 1]]], dtype=torch.float32)
print(x.max(dim=1)[0])
print(x.mean(dim=1))
'''

''' test tensor.repeat() '''
x = torch.tensor([[1, 1], [2, 2], [3, 3]])
dim = x.size(1)
y = x.repeat(1, 3)
print(y.view(-1, dim))
print(x)


''' test torch.nn.functional.softmax
x = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=torch.float32)
print(torch.nn.functional.softmax(x, dim=1))
'''

''' test torch.matmul
x = torch.randn((10, 3, 4))
y = torch.randn((10, ))
'''

''' test torch.squeeze()
x = torch.tensor([[[1, 2]], [[1, 2]]])
print(x.squeeze_(1))
'''

''' test torch.add()
x = torch.tensor([[1, 2, 3], [2, 3, 6]])
y = torch.tensor([[1, 1, 1], [4, 4, 4]])
print(torch.add(x, y))
'''

''' test tensor index  && torch.shuffle()
x = torch.tensor([[1, 2, 3], [2, 3, 6], [4, 5, 6]])
y = torch.tensor([2, 1])
z = x[y]
z = z[0: 2, 0: 2]
print(z)
'''

''' test normalizion by numpy and pytorch
x = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 4, 2]], dtype=np.float32)
l2_x = np.linalg.norm(x, axis=1, keepdims=True)
norm_x = x / l2_x
print(norm_x)

x = torch.tensor(x)
print(torch.nn.functional.normalize(x, p=2, dim=1))
'''

''' test string.strip().split()
str1 = '  1 2 3 4  5 6 7 8 9 0'
str1 = [int(s) for s in str1.strip().split()]
str2 = [0]
print(str2.extend(str1))
print(str2)
'''

''' torch.nn.functional.cross_entropy 
input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
print(target)
print(torch.nn.functional.cross_entropy(input, target, reduction='none'))
'''

'''
print(np.zeros(shape=(5)))
print(np.zeros(shape=(5, 1)))
'''

''' test torch.norm
x = torch.tensor([[1, 2, 1, 1]], dtype=torch.float32)
y = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
print(x.pow(2))
print(torch.square(x))
print(torch.norm(x))
print(torch.norm(y))
'''