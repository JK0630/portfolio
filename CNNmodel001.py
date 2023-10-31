import torch
import torch.nn.init
import torchvision.datasets as dsets
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(torch.cuda.is_available())
#랜덤 시드 고정
torch.manual_seed(777)

#GPU사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed(777)
    # torch.cuda.manual_seed_all(777) # multi GPU

# a = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
# print(a.weight) # torch.manual_seed()를 사용하면 weight가 항상 동일
# print(a.bias)

learning_rate = 0.001
training_epochs = 5
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',train=True,transform=transforms.ToTensor(),download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',train=False,transform=transforms.ToTensor(),download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)

print(len(mnist_train))
print(len(mnist_test))
print(len(data_loader))

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1st layer
        # input : (batch_size, 28, 28, 1)
        # conv : (batch_size, 28, 28, 32)
        # pool : (batch_size, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            #in_channels, out_channels
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2))
        #2nd layer
        # input : (batch_size, 14, 14, 32)
        # conv : (batch_size, 14, 14, 64)
        # pool : (batch_size, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # FCL 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7*7*64,10,bias=True)
        
        # FCL weight init
        torch.nn.init.xavier_uniform_(self.fc.weight) # 자비에 초기화
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.size()) # (100, 64, 7, 7)
        out = out.view(out.size(0), -1) # Flatten
        # print(out.size()) # (100, 3136)
        out = self.fc(out)
        return out
    
# CNN 모델 정의
model = CNN().to(device)

# 비용함수로 CrossEntropyLoss() 사용
# 비용 함수에 소프트맥스 함수가 포함되어 있다.
criterion = torch.nn.CrossEntropyLoss().to(device)

# 최적화함수로 Adam사용
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0
    
    for X, Y in data_loader:  # X: mini batch(size: 100), Y: label
        # image size: 28x28
        X=X.to(device)
        Y=Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
        
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    

with torch.no_grad():# 학습을 진행하지 않음. Gradient 계산하지 않음
# 추론결과를 동일하게 하기 위해 드롭아웃과 배치정규화를 평가모드로 설정
    model.eval()
    # print(len(mnist_test), mnist_test.test_data.size()) # (10000, 28, 28)
    # view() -> (10000, 28, 28) --> (10000, 1, 28, 28)
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = model(X_test)

    # torch.argmax(prediction, 1) == Y_test -> GT와 비교해서 True, False로 리턴
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    # correct_prediction.float() -> True는 1. , False는 0.으로 리턴
    accuracy = correct_prediction.float().mean() # 모든 테스트의 결과의 평균
    # tensor 속의 값만 출력. single scalar value일때만 사용가능
    print('Accuracy:', accuracy.item() * 100 , '%') # 약 98.9 % 