import torch

y = torch.Tensor([4, 5])
loss_fn = torch.nn.MSELoss()

head1 = torch.Tensor([3])
head2 = torch.Tensor([4])
head_double = torch.cat([head1, head2])
print(head_double)

loss_single = loss_fn(head_double, y)
print(f'loss with one head: {loss_single}')

loss1 = loss_fn(head1, y[0])
loss2 = loss_fn(head2, y[1])
loss_double = (loss1 + loss2) / 2
print(f'loss with 2 heads: {loss_double}')


