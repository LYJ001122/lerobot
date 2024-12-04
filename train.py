import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.compute_stats import compute_stats
from dataset import RoboticDataset
import os
from torch.utils.tensorboard import SummaryWriter

# 특정 GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 3번 GPU만 사용하도록 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 변환
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # 이미지를 640x480 크기로 리사이즈
    transforms.ToTensor(),
])

# 데이터셋 준비
dataset = RoboticDataset(data_dir="data", transform=transform, output_steps=20)

# 데이터셋 분할 (60% train, 20% validation, 20% test)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 데이터셋 통계
dataset_stats = compute_stats(dataset)

# 모델 초기화
config = ACTConfig()  # 업데이트된 config 사용
model = ACTPolicy(config, dataset_stats=dataset_stats)  # dataset_stats 전달

# 모델을 GPU로 이동
model = model.to(device)

# 옵티마이저 설정
optimizer = Adam(model.parameters(), lr=1e-4)

# 텐서보드 설정
log_dir = 'log_tensorboard'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
log_dir_model = 'log_model'

# 학습 루프
num_epochs = 100  # 학습할 에폭 수 설정

for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0  # 총 손실값 초기화
    
    # 학습 데이터로 학습
    for batch in train_loader:
        optimizer.zero_grad()  # 이전 기울기 초기화

        # 배치 데이터를 GPU로 이동
        batch = {k: v.to(device) for k, v in batch.items()}

        # 모델에 배치 데이터 입력하여 손실 계산
        loss_dict = model(batch)
        loss = loss_dict["loss"]  # 손실 값

        # 역전파 및 최적화
        loss.backward()  # 역전파
        optimizer.step()  # 옵티마이저를 사용하여 파라미터 업데이트

        total_loss += loss.item()  # 총 손실 값에 현재 배치 손실 더하기

    # 에폭마다 손실 출력
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

    # 텐서보드에 손실 값 기록
    writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)

    # 모델 저장 (epoch마다 저장)
    if (epoch + 1) % 1 == 0:  # 10 에폭마다 저장 (원하는 주기로 변경 가능)
        model_save_path = os.path.join(log_dir_model, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    # 검증 데이터로 검증
    model.eval()  # 모델을 평가 모드로 설정
    val_loss = 0
    with torch.no_grad():  # 검증 시에는 역전파를 하지 않음
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_dict = model(batch)
            loss = loss_dict["loss"]
            val_loss += loss.item()

    # 검증 손실 출력 및 텐서보드 기록
    print(f"Validation Loss after epoch {epoch + 1}: {val_loss / len(val_loader)}")
    writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)

# 학습 완료 후 텐서보드 종료
writer.close()
