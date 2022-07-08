# building-efficiency-utility

## Exploratory Data Analysis

## Model Development & Training 
![loss plot](./results/loss_plot.png)
train loss: 7.910704612731934 val loss: 1.6574316024780273 test loss: 3.789883613586426


Model(
  (linear_stack): Sequential(
    (0): Linear(in_features=8, out_features=20, bias=True)
    (1): ReLU()
    (2): Linear(in_features=20, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=20, bias=True)
    (5): ReLU()
    (6): Linear(in_features=20, out_features=20, bias=True)
    (7): ReLU()
    (8): Linear(in_features=20, out_features=20, bias=True)
    (9): ReLU()
    (10): Linear(in_features=20, out_features=2, bias=True)
  )
)

x_train: (621, 8)
y_train: (621, 2)
x_val: (70, 8)
y_val: (70, 2)
x_test: (77, 8)
y_test: (77, 2)
