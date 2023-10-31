import torch
import torch.nn.init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
        super(CNN, self).__init__() # super().__init__() : 부모클래스 초기화
        self.keep_prob = 0.5
        # 1st layer
        # input : (batch_size, 28, 28, 1)
        # conv : (batch_size, 28, 28, 32)
        # pool : (batch_size, 14, 14, 32)
        self.layer1 = torch.nn.Sequential( 
        # in_channels, out_channels
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 2nd layer
        # input : (batch_size, 14, 14, 32)
        # conv : (batch_size, 14, 14, 64)
        # pool : (batch_size, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Layer 3 input img shape=(batch_size, 7, 7, 64) 
        # Conv ->(batch_size, 7, 7, 128) 
        # Pool ->(batch_size, 4, 4, 128) 
        self.layer3 = torch.nn.Sequential( 
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)) 
        # Layer 4 FCL 4x4x128 inputs -> 625 outputs 
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True) 
        torch.nn.init.xavier_uniform_(self.fc1.weight) 
        self.layer4 = torch.nn.Sequential( 
            self.fc1, 
            torch.nn.ReLU(), 
            torch.nn.Dropout(p=1 - self.keep_prob)) 
        # Layer 5 FCL 625 inputs -> 10 outputs 
        self.fc2 = torch.nn.Linear(625, 10, bias=True) 
        torch.nn.init.xavier_uniform_(self.fc2.weight) 

    def forward(self, x): 
        out = self.layer1(x) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = out.view(out.size(0), -1) # Flatten them for FC 
        out = self.layer4(out) 
        out = self.fc2(out) 
        return out



model = CNN().to(device)
if 1:
    model = torch.load('test.pt')
    with torch.no_grad(): # 학습을 진행하지 않을 것이므로 torch.no_grad(), 
        #gradient 계산하지 않음
        model.eval()
        
        # print(len(mnist_test))
        X_test = mnist_test.test_data[1120].view(1, 1, 28, 28).float().to(device)
        Y_test = mnist_test.test_labels[1120].to(device) 
        
        print(X_test[0].size(), Y_test.item())
        
        prediction = model(X_test)
        print(torch.argmax(prediction, 1).item(), Y_test.item())
        # img = X_test[0].permute(1, 2, 0).numpy()
        plt.imshow(X_test[0].permute(1, 2, 0))
