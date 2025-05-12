import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

loaded_data = np.load('csi_scaled_train_test.npz', allow_pickle=True)

X_train = loaded_data['X_train_scaled'] #특징값
y_train = loaded_data['y_train'] #라벨값
y_train = 2 * y_train - 1 # 라벨링을 -1과 1로 바꿔줌 (SVM의 hinge loss 때문에)

X_test = loaded_data['X_test_scaled']   #특징값
y_test = loaded_data['y_test']   #라벨값
y_test[y_test == 0] = -1 # 라벨링을 -1과 1로 바꿔줌 (SVM의 hinge loss 때문에)

#numpy -> torch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

#모델 정의
input_dim = X_train.shape[1] #210

class SVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fully_connected = nn.Linear(input_dim,1) # nn.Linear(입력개수,출력개수)
    
    def forward(self,x):
        return self.fully_connected(x)

#hinge loss 정의
def hinge_loss(outputs, targets):
    targets = targets.view(-1,1)
    return torch.mean(torch.clamp(1 - outputs * targets, min=0))

#하이퍼파라미터 및 모델 초기화
model = SVM(input_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001) #lr = 0.1->0.01 weight_decay = 0.1 -> 0.001 하니깐 정확도 4% 증가
num_epoch = 20 #epoch 10->20으로 올리니깐 .24% 올랐다.
batch_size = 1

#학습 루프
for epoch in range(num_epoch):
    permutation = torch.randperm(X_train.size()[0])
    epoch_loss = 0
    
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        outputs = model(batch_x)
        loss = hinge_loss(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

#평가
with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.sign(outputs).squeeze()
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item()*100:.2f}%")