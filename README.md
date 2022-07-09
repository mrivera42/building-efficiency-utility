# building-efficiency-utility

Predicting energy performance of buildings (EPB) is important for designing buildings that minimize energy waste and adverse impacts on the environment.
In this projects I developed a PyTorch neural network that predicts energy loads of buildings based on 8 input characteristics. The model was then deployed in a web app using a REST API. The application was then containerized with Docker. 

## Exploratory Data Analysis
The dataset consisted of 768 samples, 8 features, and 2 outputs (heating load, and cooling load), with the units shown below:
* relative compactness (decimal) 
* surface area (m^2)
* wall area (m^2)
* roof area (m^2)
* overall height (m)
* orientation (2,3,4, or 5)
* glazing area (decimal percentage)
* glazing area distribution (0-5)
  * 0 = no glazing distribution (no glazing area) 
  * 1 = uniform: with 25% glazing on each size 
  * 2 = north: 55% on the north side and 15% on the other sides 
  * 3 = east: 55% on the east side and 15% on each of the other sides 
  * 4 = south: 55% on the south side and 15% on each side of the other sides 
  * 5 = west: 55% on the west side and 15% on each of the other sides 

A distribution of all the features was plotted. 
![features](./results/features.png)

The correlation of each feature with the heating load and cooling load was plotted. 
![heating correlation](./results/heating_correlation.png)
![cooling correlation](./results/cooling_correlation.png)
The correlation of each feature with eachother was plotted. 
![feature correlation](./results/corr_plot.png)

## Model Development & Training 
Since we have 2 outputs that are continuous variables, the the problem can be modeled as a muli-task regression. Using PyTorch, a model with the following structure
was created: 
![model structure](./results/model_structure.png)
The dataset was split into a training, validation, and test set with the following shapes: 
```
x_train: (621, 8)
y_train: (621, 2)
x_val: (70, 8)
y_val: (70, 2)
x_test: (77, 8)
y_test: (77, 2)
```
Since this is a regression problem, MSE loss is used. But how can MSELoss be used for a multi-task problem? One solution is to create 2 output heads in the network, compute MSELoss for both, then sum or average them. Luckily, PyTorch's torch.nn.MSELoss() can take in vectors and averages the MSELoss for each of their values for us. 
The model was trained for 20 epochs using the Adam optimizer, yielding a final training loss of 7.910704612731934, validation loss of 1.6574316024780273, and test loss of 3.789883613586426. 
![loss plot](./results/loss_plot.png)

## Model Deployment 


## Containerization 


