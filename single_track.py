import torch
import pandas as pd
import torch.nn as nn
import numpy as np
#class classname(nn.Module)

# initialise parameters
# cost 
# backpropogation
# update parameters

class PhysicsInformedModel(nn.Module):
    def __init__(self, input_dim, channels_depths, output_dim, act=nn.ReLU):
        super(PhysicsInformedModel, self).__init__()
        layers = list()
        layers.append(nn.Sequential(nn.Linear(input_dim,channels_depths[0]), act()))
        for k, _ in enumerate(channels_depths[:-1]):
            layers.append(nn.Sequential(nn.Linear(channels_depths[k],channels_depths[k+1]), act()))
        layers.append(nn.Linear(channels_depths[-1],output_dim ))
        self.layers = nn.Sequential(*layers)
        self.wheelbase = nn.Parameter(torch.tensor(1.0, dtype=torch.float)) # parameter to identify


        self.mse = nn.MSELoss()
        
    def forward(self,X):
        return self.layers(X)

    def get_loss(self, X):
        prediction = self.forward(X)
        prediction_with_measurements = prediction[:, 2:]                       
        fitting_loss = self.mse(prediction_with_measurements, X[:, 1:])
        
        (x, y, v, orientation, acceleration, steering_angle) = torch.split(prediction, split_size_or_sections=1, dim=1)
        x_dot = torch.autograd.grad(prediction[:,0].sum(dim=0), X, create_graph=True)[0][:,:1]
        y_dot = torch.autograd.grad(prediction[:,1].sum(dim=0), X, create_graph=True)[0][:,:1]
        v_dot = torch.autograd.grad(prediction[:,2].sum(dim=0), X, create_graph=True)[0][:,:1]
        orientation_dot = torch.autograd.grad(prediction[:,3].sum(dim=0), X, create_graph=True)[0][:,:1]

        ode_loss = 0
        ode_loss += self.mse(x_dot, v * torch.cos(orientation))
        ode_loss += self.mse(y_dot, v * torch.sin(orientation))
        ode_loss += self.mse(v_dot, acceleration)
        ode_loss += self.mse(orientation_dot, v / self.wheelbase * torch.tan(steering_angle))

        return fitting_loss + ode_loss

# prediction = [x_pos, y_pos, v, orientation, acceleration, steering_angle, xdot, ydot...]


# loss_module = Loss()
# for data in dataset:
#     out = net(data)
#     loss = loss_module(out, data)
#     loss.backward()

class SingleTrackData():
    def __init__(self):
        dfs = [pd.read_csv(f'main/Data/generated_data/{k:05d}.csv', sep=',',header=0) for k in range(1000)]
        self.velocity_star = np.concatenate([np.array(df['velocity']) for df in dfs])
        self.orientation_star = np.concatenate([np.array(df['orientation']) for df in dfs])
        self.steering_angle_star = np.concatenate([np.array(df['steering wheel angle']) for df in dfs])
        self.acceleration_star = np.concatenate([np.array(df['acceleration']) for df in dfs])
        self.time_star = np.concatenate([np.array(df['time']) for df in dfs])

    def __getitem__(self, idx):
        return torch.tensor(np.array([
                        self.time_star[idx],
                        self.velocity_star[idx], 
                        self.orientation_star[idx],
                        self.steering_angle_star[idx],
                        self.acceleration_star[idx]
                        ]), dtype=torch.float, requires_grad=True)

    def __len__(self):
        return self.time_star.shape[0]
    

if __name__ == "__main__":
    # Load Data
    import torch.utils.data as data
    dataset = SingleTrackData()
    dataloader = data.DataLoader(dataset, 10, shuffle=True)
    model = PhysicsInformedModel(input_dim=5,channels_depths=[10,10],output_dim=6).float()

    test_data = next(iter(dataloader))
    loss = model.get_loss(test_data)
    print(loss)
    