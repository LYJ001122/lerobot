import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.compute_stats import compute_stats
from dataset import RoboticDataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CUDA(=GPU) 사용 가능 여부 확인

# 이미지 변환
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # 이미지를 640x480 크기로 리사이즈
    transforms.ToTensor(),
])

# 데이터셋과 DataLoader
dataset = RoboticDataset(data_dir="data", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

dataset_stats = compute_stats(dataset)

config = ACTConfig()  # 업데이트된 config 사용
model = ACTPolicy(config, dataset_stats=dataset_stats)  # dataset_stats 전달

# 모델을 GPU로 이동
model = model.to(device)

optimizer = Adam(model.parameters(), lr=1e-4)

# 5. 학습 루프
num_epochs = 100  # 학습할 에폭 수 설정

for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0  # 총 손실값 초기화
    
    for batch in dataloader:
        optimizer.zero_grad()  # 이전 기울기 초기화

        # 배치 데이터를 GPU로 이동
        batch = {k: v.to(device) for k, v in batch.items()}

        # 6. 모델에 배치 데이터 입력하여 손실 계산
        loss_dict = model(batch)
        loss = loss_dict["loss"]  # 손실 값

        # 7. 역전파 및 최적화
        loss.backward()  # 역전파
        optimizer.step()  # 옵티마이저를 사용하여 파라미터 업데이트

        total_loss += loss.item()  # 총 손실 값에 현재 배치 손실 더하기

    # 에폭마다 손실 출력
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}")