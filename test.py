import time
import os
import cv2
import numpy as np
from dynamixel_sdk import *  # Dynamixel SDK 사용
import pyrealsense2 as rs  # Intel RealSense 라이브러리
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class ACTPolicyNetwork(nn.Module):
    def __init__(self):
        super(ACTPolicyNetwork, self).__init__()
        # CNN for image feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # Fully connected layers for combining features
        self.fc1 = nn.Linear(32 * 37 * 27 + 6, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)  # Output next motor angles

    def forward(self, image, motor_angles):
        image_features = self.cnn(image)
        combined = torch.cat((image_features, motor_angles), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def set_motors_torque(motor_ids, enable):
    for motor_id in motor_ids:
        result, error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, enable)
        if result != COMM_SUCCESS:
            print(f"모터 {motor_id} 토크 설정 실패: {packetHandler.getTxRxResult(result)}")
        elif error != 0:
            print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")

def set_current_based_position_mode(motor_id):
    OPERATING_MODE_ADDR = 11  # Operating Mode 주소
    CURRENT_BASED_POSITION_MODE = 5  # Current-Based Position 모드

    # Torque Disable (Operating Mode를 설정하려면 토크를 비활성화해야 함)
    result, error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if result != COMM_SUCCESS:
        print(f"모터 {motor_id} 토크 비활성화 실패: {packetHandler.getTxRxResult(result)}")
        return False
    elif error != 0:
        print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")
        return False

    # Operating Mode 설정
    result, error = packetHandler.write1ByteTxRx(portHandler, motor_id, OPERATING_MODE_ADDR, CURRENT_BASED_POSITION_MODE)
    if result != COMM_SUCCESS:
        print(f"모터 {motor_id} Operating Mode 설정 실패: {packetHandler.getTxRxResult(result)}")
        return False
    elif error != 0:
        print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")
        return False

    print(f"모터 {motor_id}가 Current-Based Position 제어로 변경되었습니다.")
    return True

def set_motor_current(motor_id, current):
    CURRENT_CONTROL_ADDR = 102  # Goal Current 주소

    # 전류 값 설정 (32비트)
    result, error = packetHandler.write2ByteTxRx(portHandler, motor_id, CURRENT_CONTROL_ADDR, current)
    if result != COMM_SUCCESS:
        print(f"모터 {motor_id} Current 설정 실패: {packetHandler.getTxRxResult(result)}")
        return False
    elif error != 0:
        print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")
        return False

    print(f"모터 {motor_id}의 Current가 {current}로 설정되었습니다.")
    return True


def read_motor_position(motor_id):
    position, result, error = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
    if result != COMM_SUCCESS:
        print(f"모터 {motor_id} 각도 읽기 실패: {packetHandler.getTxRxResult(result)}")
    elif error != 0:
        print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")
    return position

def read_motors_positions(groupBulkRead, motor_ids):
    motor_positions = []

    # Bulk Read 수행
    result = groupBulkRead.txRxPacket()
    if result != COMM_SUCCESS:
        print("Bulk Read 실패:", packetHandler.getTxRxResult(result))
        return motor_positions  # 빈 딕셔너리 반환

    # 각 모터의 위치 데이터를 읽음
    for motor_id in motor_ids:
        if groupBulkRead.isAvailable(motor_id, ADDR_PRESENT_POSITION, 4):
            motor_positions.append(groupBulkRead.getData(motor_id, ADDR_PRESENT_POSITION, 4))
        else:
            print(f"모터 {motor_id} 데이터 읽기 실패")
    if motor_positions[-1] > 2048:
        motor_positions[-1] = 2560
    else :
        motor_positions[-1] = 1900
    return motor_positions

def set_motor_position(motor_id, position):
    result, error = packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, position)
    if result != COMM_SUCCESS:
        print(f"모터 {motor_id} 목표 각도 설정 실패: {packetHandler.getTxRxResult(result)}")
    elif error != 0:
        print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")

def set_multiple_motor_positions(groupSyncWrite, motor_ids, positions):
    for motor_id, position in zip(motor_ids, positions):
        # 목표 위치를 4바이트 데이터로 변환 (Little Endian)
        param_goal_position = [
            DXL_LOBYTE(DXL_LOWORD(position)),  # 하위 2바이트
            DXL_HIBYTE(DXL_LOWORD(position)),  # 상위 2바이트 (하위 워드)
            DXL_LOBYTE(DXL_HIWORD(position)),  # 상위 2바이트 (상위 워드)
            DXL_HIBYTE(DXL_HIWORD(position)),  # 상위 2바이트
        ]
        
        # GroupSyncWrite에 모터 추가
        if not groupSyncWrite.addParam(motor_id, param_goal_position):
            print(f"모터 {motor_id}의 목표 위치 추가 실패")
            return False

    # 설정된 모든 모터에 데이터 전송
    dxl_comm_result = groupSyncWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"GroupSyncWrite 전송 실패: {packetHandler.getTxRxResult(dxl_comm_result)}")
        return False

    # GroupSyncWrite에서 데이터 초기화
    groupSyncWrite.clearParam()  # 데이터 전송 후에는 반드시 초기화
    return True

def add_motor_to_bulk_read(groupBulkRead, motor_ids):
    for motor_id in motor_ids:
        # ADDR_PRESENT_POSITION: 읽고자 하는 데이터의 시작 주소
        # 4: 데이터 길이 (현재 위치는 4바이트 데이터)
        if not groupBulkRead.addParam(motor_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
            print(f"모터 {motor_id}를 Bulk Read에 추가하는 데 실패했습니다.")

def set_profile_velocity(motor_ids, velocity):
    for motor_id in motor_ids:
        # Profile Velocity 설정
        result, error = packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_VELOCITY, velocity)
        if result != COMM_SUCCESS:
            print(f"모터 {motor_id} Profile Velocity 설정 실패: {packetHandler.getTxRxResult(result)}")
        elif error != 0:
            print(f"모터 {motor_id} 통신 오류 발생: {packetHandler.getRxPacketError(error)}")
        else:
            print(f"모터 {motor_id} Profile Velocity가 {velocity}로 설정되었습니다.")

import os

def get_largest_first_number(folder_path):
    """
    폴더 내 숫자로 된 이름을 가진 하위 폴더 중 가장 큰 숫자를 반환합니다.
    폴더가 없거나 숫자가 없으면 0을 반환합니다.
    """
    # 폴더 내 하위 폴더 이름 확인
    existing_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    numbers = []
    
    for folder_name in existing_folders:
        try:
            # 폴더 이름을 숫자로 변환
            numbers.append(int(folder_name))
        except ValueError:
            # 폴더 이름이 숫자가 아닌 경우 무시
            continue

    # 가장 큰 숫자 반환 (없으면 0 반환)
    return max(numbers) if numbers else 0




def save_position(data, file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    np.save(file_path, data)
    return 


def save_image(image, file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)

    cv2.imwrite(file_path, image)
    return 

def create_folder_if_not_exists(folder_path):
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_motor_angles(model, image, motor_angles):
    # 이미지 전처리
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 후 GPU로 이동
    motor_angles = torch.tensor(motor_angles, dtype=torch.float32).unsqueeze(0).to(device)  # 모터 각도를 텐서로 변환
    
    with torch.no_grad():  # Gradients 계산하지 않도록 설정
        next_motor_angles = model(image, motor_angles)  # 모델 예측
    
    # 예측된 값을 int로 변환 후 반환
    return next_motor_angles.squeeze().cpu().numpy().astype(int)  # int로 변환




# 다이나믹셀 제어에 필요한 설정
DEVICENAME = '/dev/dynamixel'  # USB 포트 (Windows는 'COMX' 형식, Linux는 '/dev/ttyUSBX' 형식)
BAUDRATE = 57600  # 통신 속도
PROTOCOL_VERSION = 2.0  # Dynamixel 프로토콜 버전
ADDR_PRESENT_POSITION = 132  # 현재 포지션 값을 읽어올 주소
ADDR_GOAL_POSITION = 116  # 목표 포지션을 설정할 주소
ADDR_TORQUE_ENABLE = 64  # 토크 설정 주소
ADDR_PROFILE_VELOCITY = 112
LEN_PRESENT_POSITION = 4  # 포지션 데이터 길이
LEN_GOAL_POSITION = 4
TORQUE_ENABLE = 1  # 토크 활성화
TORQUE_DISABLE = 0  # 토크 비활성화
FPS = 20  # 프레임 레이트 (30 FPS)
PERIOD = 1.0 / FPS  # 한 프레임의 시간
RESOLUTION = (640, 480)
TIME_LENGTH = 20
GRIPPER_CURRENT = 70

# 모터의 ID와 허용된 각도 범위 설정
motor_ids_follower = [10, 11, 12, 13, 14, 15]
init_positions_follower = [2048, 2048, 2048, 2048, 2048, 2048]

position_limits = {
    0: (1, 4094),
    1: (1390, 2480),
    2: (1024, 3200),
    3: (1770, 4094),
    4: (50, 4055),
    5: (1, 4094),
    10: (1, 4094),
    11: (1390, 2480),
    12: (1024, 3200),
    13: (1770, 4094),
    14: (50, 4055),
    15: (1, 4094)
}

# 모델 불러오기 (이미지와 6개의 모터 값을 입력받아 예측)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ACTPolicyNetwork()
model.load_state_dict(torch.load("log/act_policy_model_epoch_4.pth"))
model.to(device)
model.eval()  # Inference 모드로 설정

# 모델 입력을 위한 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((120, 160)),  # 모델에 맞는 크기로 조정
])

# 포트와 패킷 핸들러 설정
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
groupBulkRead = GroupBulkRead(portHandler, packetHandler)
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

folder_path = "data"
folder_number = get_largest_first_number(folder_path) + 1

# 포트를 열고 통신 속도를 설정
if portHandler.openPort():
    print("포트 열기 성공")
else:
    print("포트 열기 실패")
    quit()

if portHandler.setBaudRate(BAUDRATE):
    print("통신 속도 설정 성공")
else:
    print("통신 속도 설정 실패")
    quit()

set_current_based_position_mode(motor_ids_follower[-1])
set_motor_current(motor_ids_follower[-1], GRIPPER_CURRENT)
add_motor_to_bulk_read(groupBulkRead, motor_ids_follower)

# RealSense 카메라 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, 30)

# 스트림 시작
pipeline.start(config)

# 팔로워 모터의 토크 활성화
set_motors_torque(motor_ids_follower, TORQUE_ENABLE)

set_profile_velocity(motor_ids_follower, 10)
set_multiple_motor_positions(groupSyncWrite, motor_ids_follower, init_positions_follower)
set_profile_velocity(motor_ids_follower, 30)

try:
    while True:
        key = cv2.waitKey(1) & 0xFF

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow("RealSense", color_image)

        if key == ord('q'):  # 'q' 키를 누르면 종료
            break
        elif key == ord('s'):
            while True:
                start_time = time.time()
            
                positions = read_motors_positions(groupBulkRead, motor_ids_follower)

                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                #     cv2.imshow("RealSense", color_image)

                next_positions = get_motor_angles(model, color_image, positions)
                set_multiple_motor_positions(groupSyncWrite, motor_ids_follower, next_positions)

                time_use = time.time() - start_time
                if PERIOD < time_use:
                    print(time_use)
                time.sleep(max(0, PERIOD - time_use))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    
            time.sleep(3)

            set_profile_velocity(motor_ids_follower, 10)
            set_multiple_motor_positions(groupSyncWrite, motor_ids_follower, init_positions_follower)
            set_profile_velocity(motor_ids_follower, 30)

                
except KeyboardInterrupt:
    print("종료")

finally:
    # 팔로워 모터의 토크 비활성화
    set_motors_torque(motor_ids_follower, TORQUE_DISABLE)

    # RealSense 종료
    pipeline.stop()

    # 포트 닫기
    portHandler.closePort()
    cv2.destroyAllWindows()

