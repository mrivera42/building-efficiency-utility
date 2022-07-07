import torch 
import pandas as pd
from sklearn.model_selection import KFold

class Model(torch.nn.Module):
    def __init__(self, num_features):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(num_features,20)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20,20)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(20,20)
        self.activation3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(20,2)
    
    def forward(self,inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.linear4(x)
        return x
        


if __name__ == '__main__':

    # load the data 
    df = pd.read_csv('energy_efficiency_data.csv')

    print(df.shape)
    print(df.head())

    # remove head 
    headers = list(df.columns)
    print(headers)
