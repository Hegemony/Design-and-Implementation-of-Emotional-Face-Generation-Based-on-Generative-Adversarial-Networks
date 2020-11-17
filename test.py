# import pickle
# pkl_path = 'aus_openface.pkl'
# saved_dict = {}
# with open(pkl_path, 'rb') as f:
#     saved_dict = pickle.load(f, encoding='latin1')
#
# print(saved_dict)
import torch

x = torch.Tensor([[1], [2], [3]])
y = torch.Tensor([[[1], [2], [3]]])
z = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
print(x.size())
print(y.size())
print(z.size())
print(x.expand(3, 2))
print(x.expand(3, 2).size())
print(z.expand(1, 2, 4))
print(z.expand(4, 2, 4).size())
print(x.repeat(1, 4))
print(x.repeat(1, 4).size())
print(y.expand(1, 3, 2))
print(y.repeat(1, 2, 5))

class Person(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
    def __call__(self, friend):
        print('My name is %s...' % self.name)
        print('My friend is %s...' % friend)
p = Person('Bob', 'male')
print(p('Tim'))



