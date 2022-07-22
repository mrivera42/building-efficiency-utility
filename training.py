import torch 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


# model class 
class Model(torch.nn.Module):
    def __init__(self, num_features, num_hidden_layers, num_units):
        super(Model,self).__init__()

        self.linear_in = torch.nn.Linear(num_features, num_units)
        hidden_layers = []
        for i in range(num_hidden_layers): 
            hidden_layers.append(torch.nn.Linear(num_units, num_units))
            hidden_layers.append(torch.nn.ReLU())
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.linear_out = torch.nn.Linear(num_units, 2)
    
    def forward(self,inputs):
        x = self.linear_in(inputs)
        x = self.hidden_layers(x)
        x = self.linear_out(x)
        return x




            

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self,x,y):
        self.x = x
        self.y = y 
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return len(self.x)
    
        


if __name__ == '__main__':

    # check GPU
    if torch.backends.mps.is_available():
        print('device set to MPS')
        device = torch.device('mps')
    else:
        print('MPS not available, device set to CPU')
        device = torch.device('mps')

    

    # load the data 
    df = pd.read_csv('energy_efficiency_data.csv')

    # split data into train and test 
    x = df.iloc[:,0:-2]
    y = df.iloc[:,-2:]
    x_trainval, x_test, y_trainval, y_test = train_test_split(x,y,test_size=0.1)


    # hyperparameters 
    hyperparams = {
        'lr': [1e-2,1e-3, 1e-4],
        'num_hidden_layers': [3,4,5],
        'num_units' : [15,20,25],
        'num_epochs': [10, 20, 30]
    }
    k_folds = 5


    for lr in hyperparams['lr']: 
        for num_hidden_layers in hyperparams['num_hidden_layers']: 
            for num_units in hyperparams['num_units']: 
                for num_epochs in hyperparams['num_epochs']:

                    # create model 
                    name = f'lr_{lr}_layers_{num_hidden_layers}_units_{num_units}_epochs_{num_epochs}'
                    model = Model(
                        num_features=8,
                        num_hidden_layers=num_hidden_layers,
                        num_units=num_units
                    )

                    print(name)

                    # cross validation 
                    kfold = KFold(n_splits=k_folds, shuffle=True)
                    for fold, (train_ids, val_ids) in enumerate(kfold.split(x_trainval)): 

                        # normalize data 
                        



                    







    # x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1)

    # print(f'x_train: {x_train.shape}')
    # print(f'y_train: {y_train.shape}')
    # print(f'x_val: {x_val.shape}')
    # print(f'y_val: {y_val.shape}')
    # print(f'x_test: {x_test.shape}')
    # print(f'y_test: {y_test.shape}')


    # # normalize data
    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)

    # # convert everything to numpy arrays 
    # x_train, y_train = np.array(x_train), np.array(y_train)
    # x_val, y_val = np.array(x_val), np.array(y_val)
    # x_test, y_test = np.array(x_test), np.array(y_test)

    # # convert type to float 
    # y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

    # # create datasets 
    # ds_train = CustomDataset(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float())
    # ds_val = CustomDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
    # ds_test = CustomDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
    # # ds_train = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    # # ds_val = torch.utils.data.TensorDataset(x_val, y_val)
    # # ds_test = torch.utils.data.TensorDataset(x_test, y_test)
    
    # # wrap in batches 
    # batch_size=32
    # dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size)
    # dl_val = torch.utils.data.DataLoader(dataset=ds_val, batch_size=batch_size)
    # dl_test = torch.utils.data.DataLoader(dataset=ds_test, batch_size=batch_size)


    # # training loop 
    # epochs = 20 
    # model = Model(8)
    # model.to(device)
    # print(model)

    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    # stats = {
    #     'train_loss': [], 
    #     'val_loss': []
    # }

    # for epoch in range(1, epochs + 1): 

    #     train_size = len(dl_train.dataset)
    #     total_loss = 0
        
    #     for batch_idx, (x, y) in enumerate(dl_train): 

    #         x = x.to(device)
    #         y = y.to(device)

    #         # forward pass 
    #         pred = model(x)

    #         # compute loss 
    #         train_loss = loss_fn(pred, y)

    #         # backprop 
    #         optimizer.zero_grad()
    #         train_loss.backward()

    #         # update parameters
    #         optimizer.step()

    #     # validation loop 
    #     num_batches = len(dl_val)
    #     with torch.no_grad():

    #         for x, y in dl_val: 
    #             x = x.to(device)
    #             y = y.to(device)

    #             # forward 
    #             pred = model(x)

    #             # compute loss 
    #             val_loss = loss_fn(pred, y)
        
    #             val_loss /= num_batches

            

    #     train_loss = train_loss.item()
    #     val_loss = val_loss.item()
    #     stats['train_loss'].append(train_loss)
    #     stats['val_loss'].append(val_loss)
    #     print(f'[{epoch}/{epochs}] train loss: {train_loss} val loss: {val_loss}')
    

    # # plot loss 
    # fig = plt.figure(figsize=(10,10))
    # plt.plot(stats['train_loss'],label='train_loss')
    # plt.plot(stats['val_loss'],label='val_loss')
    # plt.title('Loss Plot')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE loss')
    # plt.legend()
    # plt.savefig('results/loss_plot.png')


    # # model evaluation 
    # test_size = len(dl_test)

    # with torch.no_grad():

    #     for x, y in dl_test: 

    #         x = x.to(device)
    #         y = y.to(device)

    #         # forward 
    #         pred = model(x)
            
    #         # compute loss
    #         test_loss=loss_fn(pred, y)

    
    # test_loss /= test_size
    # print(f'Model Evaluation - Loss: {test_loss}')

    # model.to(torch.device('cpu'))
    # model_scripted = torch.jit.script(model)
    # model_scripted.save('model_scripted.pt')

    

            

            

            

            

    


    