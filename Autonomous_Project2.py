import torch
import torch.nn as nn
import torch.optim as optim
import random

#Training data
inputs = []
outputs = []
for i in range(2000): #could increase for better accuracy
    x1 = random.uniform(0,10) #need random float coordinates
    y1 = random.uniform(0,10)
    x2 = random.uniform(0,10)
    y2 = random.uniform(0,10)

    distance = (((x2 - x1)**2) + ((y2 - y1)**2)) ** 0.5 

    inputs.append([x1, y1, x2, y2])
    outputs.append([distance])

inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

class DistancePredictor(nn.Module):
    #defining the neural network
    def __init__(self):
        super(DistancePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), #might be too many neurons
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
            return self.net(x)

        

model1 = DistancePredictor()


#Training model
criterion = nn.MSELoss()
optimizer = optim.Adam(model1.parameters(), lr = 0.01)

epoch = 10000 #could increase for smaller deviations
for i in range(epoch):
    optimizer.zero_grad()
    predictions = model1(inputs)
    loss = criterion(predictions, outputs)
    loss.backward()
    optimizer.step()
    if i % 200 == 0:
        print(f"Epoch {i}, Loss: {loss.item():.4f}")

#Check if model works

input1 = torch.tensor([[9.0, 5.0, 4.0 , 8.0]])
prediction = model1(input1).item()
print(f"my model predicts: {prediction}")
print(f"the actual number should be 5.83")


        