import torch
import torch.nn.init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# GPU 사용이 가능한 경우 GPU를, 그렇지 않으면 CPU를 사용하도록 설정합니다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 결과의 재현성을 위해 랜덤 시드를 설정합니다.
torch.manual_seed(777)

# GPU를 사용하는 경우, GPU의 랜덤 시드도 설정합니다.
if device == 'cuda':
    torch.cuda.manual_seed(777)

# 하이퍼파라미터 설정
learning_rate = 0.001
training_epochs = 5
batch_size = 100

# MNIST 훈련 데이터셋을 다운로드하고 불러옵니다.
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)

# MNIST 테스트 데이터셋을 다운로드하고 불러옵니다.
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

# 배치 크기대로 데이터를 쉽게 불러올 수 있게 DataLoader를 생성합니다.
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# 데이터셋과 DataLoader의 길이를 출력하여 확인합니다.
print(len(mnist_train))
print(len(mnist_test))
print(len(data_loader))

# CNN 모델 구조를 정의합니다.
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.keep_prob = 0.5  # 드롭아웃을 위한 확률 설정
        
        # 첫 번째 계층: 합성곱 + ReLU + 최대 풀링
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 합성곱 계층
            torch.nn.ReLU(),  # 활성화 함수
            torch.nn.MaxPool2d(kernel_size=2, stride=2))  # 풀링 계층
        
        # 두 번째 계층: 합성곱 + ReLU + 최대 풀링
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 세 번째 계층: 합성곱 + ReLU + 최대 풀링
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        
        # 완전 연결 계층
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)  # Xavier 초기화를 사용하여 가중치 초기화
        
        # ReLU 활성화 함수와 드롭아웃 적용
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        
        # 최종 완전 연결 계층으로 클래스 점수(0-9의 숫자에 대한 10개의 클래스)를 출력합니다.
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # 네트워크를 통한 전방전파를 정의합니다.
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # 완전 연결 계층을 위해 출력을 평탄화합니다.
        out = self.layer4(out)
        out = self.fc2(out)
        return out

# CNN 모델의 인스턴스를 생성하고 해당 디바이스로 이동시킵니다.
model = CNN().to(device)

# 사전 훈련된 모델이 있는 경우 불러옵니다.
if 1:
    model = torch.load('test.pt')
    
    # 평가 모드로 모델을 전환합니다 (드롭아웃 및 배치 정규화 비활성화).
    model.eval()
    
    # 테스트 샘플을 추출합니다.
    X_test = mnist_test.test_data[1120].view(1, 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels[1120].to(device)
    
    # 모델을 사용하여 레이블을 예측합니다.
    prediction = model(X_test)
    
    # 크기를 출력하여 확인합니다.
    print(X_test[0].size(), Y_test.item())
    
    # 예측된 레이블과 실제 레이블을 출력합니다.
    print(torch.argmax(prediction, 1).item(), Y_test.item())
    
    # 테스트 이미지를 표시합니다.
    plt.imshow(X_test[0].permute(1, 2, 0))
