from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import json
from decord import VideoReader, cpu
import numpy as np
import torch.nn.functional as F

AUDIOVOLUME=30 #See utils/prepare_aria_json.py to adjust the volume of audio

def load_video(video_file):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frame_num = len(vr)
    avg_fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), avg_fps)]
    # sample_fps = round(vr.get_avg_fps() / self.data_args.video_fps)
    # frame_idx = [i for i in range(0, len(vr), sample_fps)]

    sample_fps = avg_fps

        # sample 1 frames/second
    frame_idx = [i for i in range(0, total_frame_num, sample_fps)]
    # video: (F, H, W, C)
    video = vr.get_batch(frame_idx).asnumpy()
    # Convert the list of frames to a numpy array if needed
    video = np.array(video)
    return video

# def load_audio(audio_path): #audiosegment-AUDIOVOLUME not work
#     from pydub import AudioSegment #ffmpeg
#     audiosegment = AudioSegment.from_wav(audio_path)
#     audiosegment=audiosegment-AUDIOVOLUME
#     data=np.frombuffer(audiosegment.raw_data, dtype=np.int64).reshape(-1,7)
#     print(np.array(data).shape)


def preprocess_imu(imu_paths,device,sample_mode='interpolate'):

    def _downsample(imu,sample_mode):
        if sample_mode=="interpolate":
            imu=F.interpolate(imu.unsqueeze(0).unsqueeze(0),(2000,6),mode="bilinear").squeeze(0).squeeze(0)
        else:
            indices = sorted(np.random.choice(imu.shape[0], size=2000, replace=False))
            imu = imu[indices,: ]
        return imu

    imu=[]
    totensor=lambda x:torch.tensor(x).to(device)
    for imu_path in imu_paths:
        imu_raw_data=totensor(np.load(imu_path)[:,1:])
        imu_data=_downsample(imu_raw_data,sample_mode)
        imu.append(imu_data)
    imu=torch.stack(imu, dim=0)
    imu=imu.permute(0,2,1)
    imu=imu.to(torch.float32) #Convert to float32

    
    return imu


with open("data/aria_dataset/0322_2/0322_2_result.json","r")as f:
    vrs_info=json.load(f)

image_paths=[]
audio_paths=[]
imu_left_paths=[]
imu_right_paths=[]

for clip_num,clip_info in vrs_info['video_clip'].items():
    if clip_num=="clip1" or clip_num=="clip11":

        video_path=clip_info['video']
        audio_path=clip_info['audio']
        imu_left_path=clip_info['imu_left']
        imu_right_path=clip_info['imu_right']
        gaze_info = json.load(open(clip_info['gaze'], "r"))

        image_paths.append(video_path)
        audio_paths.append(audio_path)
        imu_left_paths.append(imu_left_path)
        imu_right_paths.append(imu_right_path)


        # text_list=["A dog.", "A car", "A bird"]


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device="cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data of one clip
inputs = {
    # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION:data.load_and_transform_video_data(image_paths,device),
    ModalityType.AUDIO:  data.load_and_transform_aria_audio_data(audio_paths,device),
    ModalityType.IMU_LEFT:preprocess_imu(imu_left_paths,device,sample_mode='random'),
    ModalityType.IMU_RIGHT:preprocess_imu(imu_right_paths,device,sample_mode='random')
#     ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
