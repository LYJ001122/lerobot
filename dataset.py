import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage
from datasets import Image  # HuggingFace datasets 라이브러리의 Image 클래스 사용

class RoboticDataset(Dataset):
    def __init__(self, data_dir, transform=None, output_steps=20):
        """
        Args:
            data_dir (str): 데이터가 저장된 루트 디렉토리 경로.
            transform (callable, optional): 이미지 변환을 위한 함수.
            output_steps (int): 예측할 타임스텝의 수 (기본 20)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.output_steps = output_steps
        
        # 각 에피소드 폴더(1, 2, 3, ...) 찾기
        self.folders = sorted(os.listdir(data_dir))
        
        # 각 폴더에 대해 생성할 수 있는 데이터 샘플 수 계산
        self.samples = []
        for folder_name in self.folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            positions_path = os.path.join(folder_path, "positions.npy")
            
            # positions.npy 파일 확인
            if not os.path.exists(positions_path):
                raise FileNotFoundError(f"positions.npy not found in folder: {folder_path}")
                
            positions = np.load(positions_path)  # (seq_len, 6)
            seq_length = len(positions)

            # 한 폴더에서 생성할 수 있는 데이터 샘플 수
            folder_samples = seq_length - self.output_steps
            self.samples.extend([(folder_name, start_idx) for start_idx in range(folder_samples)])

        # features 속성 추가: 각 데이터 항목에 대한 타입 정의
        self.features = {
            "observation.images": Image(),  # datasets.Image로 이미지 타입 정의
            "observation.state": torch.float32,  # 모터 각도 (예: (6) 형태)
            "action": torch.float32,  # 출력 (예: (output_steps*6) 형태)
            "action_is_pad": torch.float32,  # 패딩 여부를 위한 필드
        }

    def __len__(self):
        # 전체 데이터 샘플 수 반환
        return len(self.samples)

    def __getitem__(self, idx):
        # 전체 샘플 리스트에서 특정 샘플 가져오기
        folder_name, start_idx = self.samples[idx]
        folder_path = os.path.join(self.data_dir, folder_name)

        # positions.npy 파일 읽기 (모터 각도)
        positions_path = os.path.join(folder_path, "positions.npy")
        positions = np.load(positions_path)  # (seq_len, 6)

        # start_idx에 해당하는 이미지 파일 읽기
        img_path = os.path.join(folder_path, f"image{start_idx}.jpg")
        if os.path.exists(img_path):
            img = PILImage.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            raise FileNotFoundError(f"Image {img_path} not found")

        # 입력은 t에서의 이미지와 모터 각도
        images_input = img  # PIL.Image 객체 (480, 640, 3)
        positions_input = positions[start_idx]  # (6)

        # 출력은 t+1부터 t+20까지의 모터 각도 (각 6개씩)
        positions_output = positions[start_idx + 1:start_idx + self.output_steps + 1]

        # 20개 타임스텝에 대한 6개 모터 각도를 일렬로 붙여서 120개의 값 생성
        # positions_output_flat = positions_output.flatten()  # (output_steps * 6)

        # 이미지를 텐서로 변환, (C, W, H) → (C, H, W) 순서로 변환
        images_input = torch.tensor(np.array(images_input).transpose(0, 2, 1), dtype=torch.float32)  # (3, 640, 480) -> (3, 480, 640)

        # 패딩이 없으므로 모든 데이터는 1로 처리 (패딩이 없다는 의미로 1을 사용)
        action_is_pad = torch.zeros(self.output_steps, dtype=torch.bool)  # 패딩이 없으므로 False로 초기화

        return {
            "observation.images": images_input,  # (3, 480, 640) 형태로 반환
            "observation.state": torch.tensor(positions_input, dtype=torch.float32),  # (6)
            "action": torch.tensor(positions_output, dtype=torch.float32),  # (output_steps * 6) 형태로 반환
            "action_is_pad": action_is_pad  # 패딩 여부 (bool 타입)
        }
