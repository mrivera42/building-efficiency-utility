import torch 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

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
    x_trainval, x_test, y_trainval, y_test = train_test_split(x,y,test_size=0.1,random_state=42)

    # normalize the test data 
    for column in x_test.columns: 
        x_test[column] = (x_test[column] - x_test[column].min()) / (x_test[column].max() - x_test[column].min())


    # hyperparameters 
    num_samples = 10
    k_folds = 5
    hyperparams_data = {
        'lr': [],
        'num_hidden_layers': [],
        'num_units': [],
        'num_epochs': [],
        'avg_training_loss': [],
        'avg_val_loss': []
    }

    for i in range(1,num_samples+1): 

        # randomly search hyperparameter space 
        lr = 10**np.random.uniform(-4,-2)
        num_hidden_layers = np.random.randint(2,10)
        num_units = np.random.randint(2,30)
        num_epochs = np.round(10**np.random.uniform(1,4)).astype(int)
        

        # create model 
        model = Model(
            num_features=8,
            num_hidden_layers=num_hidden_layers,
            num_units=num_units
        )
        model.to(device)

        # cross validation 
        train_losses = []
        val_losses = []
        kfold = KFold(n_splits=k_folds, shuffle=True)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(x_trainval)): 

            # training-validation split 
            x_train = x_trainval.iloc[train_ids, :]
            y_train = y_trainval.iloc[train_ids,:]
            x_val = x_trainval.iloc[val_ids,:]
            y_val = y_trainval.iloc[val_ids,:]
            
            # print(f'x_train: {x_train.shape}')
            # print(f'y_train: {y_train.shape}')
            # print(f'x_val: {x_val.shape}')
            # print(f'y_val: {y_val.shape}')

            # normalize data 
            x_train_new = x_train.copy()
            x_val_new = x_val.copy()
            for column in x_train.columns:
                x_train_new[column] = (x_train[column] - x_train[column].min()) / (x_train[column].max() - x_train[column].min())
                x_val_new[column] = (x_val[column] - x_val[column].min()) / (x_val[column].max() - x_val[column].min())

            # convert data to tensors 
            x_train = torch.from_numpy(np.array(x_train_new)).float()
            y_train = torch.from_numpy(np.array(y_train)).float()
            x_val = torch.from_numpy(np.array(x_val_new)).float()
            y_val = torch.from_numpy(np.array(y_val)).float()
            
            # create custom datasets 
            ds_train = CustomDataset(x_train, y_train)
            ds_val = CustomDataset(x_val, y_val)

            # create dataloaders 
            batch_size=32
            dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size)
            dl_val = torch.utils.data.DataLoader(dataset=ds_val, batch_size=batch_size)

            # print(f'dl_train: {len(dl_train)}')
            # print(f'dl_val: {len(dl_val)}')


            # start training loop 
            optimizer=torch.optim.Adam(model.parameters(),lr=lr)
            loss_fn = torch.nn.MSELoss()
            

            for epoch in range(1,num_epochs+1): 

                for batch_idx,(x,y) in enumerate(dl_train): 

                    # place tensors on device 
                    x = x.to(device)
                    y = y.to(device)

                    # forward 
                    pred = model(x)

                    # compute loss
                    loss = loss_fn(pred, y)

                    # backprop 
                    optimizer.zero_grad()
                    loss.backward()

                    # update weights
                    optimizer.step()

                train_loss = loss.item()
            
                val_loss = 0
                num_batches = len(dl_val)
                with torch.no_grad(): 

                    for x, y in dl_val: 

                        # place tensors on device
                        x = x.to(device)
                        y = y.to(device)

                        # forward
                        pred = model(x)

                        # compute loss
                        loss = loss_fn(pred, y)
                        val_loss += loss.item()
                    
                    # average val loss 
                    val_loss /= num_batches 
            
            # record the final losses for that fold 
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        # average losses over k folds 
        avg_training_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f'[{i}/{num_samples}] lr: {lr} num_hidden_layers: {num_hidden_layers} num_units: {num_units} num_epochs: {num_epochs} Avg Train Loss: {avg_training_loss} Avg Val Loss: {avg_val_loss} ')
        hyperparams_data['lr'].append(lr)
        hyperparams_data['num_hidden_layers'].append(num_hidden_layers)
        hyperparams_data['num_units'].append(num_units)
        hyperparams_data['num_epochs'].append(num_epochs)
        hyperparams_data['avg_training_loss'].append(avg_training_loss)
        hyperparams_data['avg_val_loss'].append(avg_val_loss)
    hyperparams_df = pd.DataFrame(hyperparams_data)
    hyperparams_df.to_csv('hyperparam_data.csv',index=False)
                    


    # choose the final model with the lowest val loss 
    best_values = hyperparams_df[hyperparams_df['avg_val_loss'] == hyperparams_df['avg_val_loss'].min()]
    final_lr = best_values['lr'].values[0]
    final_num_hidden_layers = best_values['num_hidden_layers'].values[0]
    final_num_units = best_values['num_units'].values[0]
    final_num_epochs = best_values['num_epochs'].values[0]
    final_loss = best_values['avg_val_loss'].values[0]
    

    print(f'lr: {final_lr} num_hidden_layers: {final_num_hidden_layers} num_units: {final_num_units} num_epochs: {final_num_epochs} has lowest val loss of {final_loss}')



                    


    

    # # plot graphs of training and val losses 
    # for model_name in stats:

    #     plt.plot(stats[model_name]['train_loss'],label=f'{model_name}')
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylable('MSE Loss')
    # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    # plt.tight_layout()
    # plt.savefig('results/hyperparam_training_lossplot.png')
    # plt.close()

    # for model_name in stats: 

    #     plt.plot(stats[model_name]['val_loss'],label=f'{model_name}')
    # plt.title('Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE Loss')
    # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    # plt.tight_layout()
    # plt.savefig('results/hyperparam_val_lossplot.png')
    # plt.close()

    
    
    # load chosen model and retrain it on all the train data
    










                        


                            

                                
                        




                        

                        


                    




                    







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

    

            

            

            

            

    


    