import torch
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from hyperparam_tuning import Model, CustomDataset
import matplotlib.pyplot as plt

if __name__ == "__main__": 

    # check GPU
    if torch.backends.mps.is_available():
        print('device set to MPS')
        device = torch.device('mps')
    else:
        print('MPS not available, device set to CPU')
        device = torch.device('mps')


    # load dataset 
    ds = pd.read_csv("energy_efficiency_data.csv")

    # split into x and y 
    x = ds.iloc[:,0:-2]
    y = ds.iloc[:,-2:]
    
    # train-test split 
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42)

    # normalize 
    x_train_norm = x_train.copy()
    x_test_norm = x_test.copy()
    y_train_norm = y_train.copy()
    y_test_norm = y_test.copy()
    for column in x_train.columns: 
        x_train_norm[column] = (x_train[column] - x_train[column].min()) / (x_train[column].max() - x_train[column].min())
        x_test_norm[column] = (x_test[column] - x_test[column].min()) / (x_test[column].max() - x_test[column].min())
    for column in y_train.columns: 
        y_train_norm[column] = (y_train[column] - y_train[column].min()) / (y_train[column].max() - y_train[column].min())
        y_test_norm[column] = (y_test[column] - y_test[column].min()) / (y_test[column].max() - y_test[column].min())
    

    # convert to numpy arrays 
    def convert_to_torch(df): 
        return torch.from_numpy(np.array(df)).float()
    x_train_norm = convert_to_torch(x_train_norm)
    x_test_norm = convert_to_torch(x_test_norm)
    y_train_norm = convert_to_torch(y_train_norm)
    y_test_norm = convert_to_torch(y_train_norm)


    # create custom dataset class 
    ds_train = CustomDataset(x_train_norm, y_train_norm)
    ds_test = CustomDataset(x_test_norm, y_test_norm)

    # wrap in batches 
    batch_size=32
    dl_train = torch.utils.data.DataLoader(dataset=ds_train, batch_size=32)
    dl_test = torch.utils.data.DataLoader(dataset=ds_test, batch_size=32)

    # set hyperparameters 
    hyperparameters_df = pd.read_csv("hyperparam_data.csv")
    best_values = hyperparameters_df[hyperparameters_df['avg_val_loss'] == hyperparameters_df['avg_val_loss'].min()]
    lr = best_values['lr'].values[0]
    num_hidden_layers = best_values['num_hidden_layers'].values[0]
    num_units = best_values['num_units'].values[0]
    # num_epochs = best_values['num_epochs'].values[0]
    num_epochs = 1

    # initialize model 
    model = Model(
        num_features=8,
        num_hidden_layers=int(num_hidden_layers),
        num_units=int(num_units)
    )
    model.to(device)

    # training loop 
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_loss = []
    for i in range(1,num_epochs+1): 

        for batch_idx, (x,y) in enumerate(dl_train): 

            # set batches to device
            x=x.to(device)
            y=y.to(device)

            # forward
            pred = model(x)

            # compute loss
            loss = loss_fn(pred, y)

            # backprop 
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # update training_loss list 
            training_loss.append(loss.item())

        print(f'[{i}/{num_epochs}] training_loss: {loss.item()}')
    
    # test loop 
    test_loss = 0
    num_batches = len(dl_test)
    with torch.no_grad(): 

        for x, y in dl_test: 

            # set tensors to device
            x = x.to(device)
            y = y.to(device)

            # forward 
            pred = model(x)

            # compute loss
            loss = loss_fn(pred,y)
            test_loss += loss.item()
        
        test_loss /= num_batches
    print(f'Test loss: {test_loss}')
    
    # export model 
    model_scripted = torch.jit.script(model)
    model_scripted.save('models/final_model.pt')

    # plot and save training loss 
    plt.plot(training_loss)
    plt.title('Final Training Loss')
    plt.xlabel('Batches')
    plt.ylabel('MSE Loss')
    plt.savefig('results/final_lossplot.png')



            

    




