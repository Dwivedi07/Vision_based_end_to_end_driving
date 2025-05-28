import tensorflow as tf
import os
import numpy as np
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

GCS_BUCKET = 'waymo_open_dataset_end_to_end_camera_v_1_0_0'  # <-- change this
MOUNT_POINT = 'gcs_bucket'      # You can rename this if needed
os.makedirs(MOUNT_POINT, exist_ok=True)

# Step 4: Mount the bucket using gcsfuse
def mount_gcs_bucket(bucket_name: str, mount_point: str):
    if not os.path.exists(mount_point):
        os.makedirs(mount_point)
    os.system(f'gcsfuse {bucket_name} {mount_point}')
mount_gcs_bucket(GCS_BUCKET, MOUNT_POINT)

GCS_BUCKET = 'waymo_open_dataset_end_to_end_camera_v_1_0_0'
# BUCKET_PATH = 'waymo_open_dataset_end_to_end_camera_v_1_0_0'

# Construct the dataset folder path using os.path.join
# DATASET_FOLDER = os.path.join('gs://', GCS_BUCKET, BUCKET_PATH)
DATASET_FOLDER = os.path.join('gs://', GCS_BUCKET)

# Match the training files using a wildcard
TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training*')
VAL_FILES = os.path.join(DATASET_FOLDER, 'val*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test*')

train_filenames = tf.io.matching_files(TRAIN_FILES)
train_dataset = tf.data.TFRecordDataset(train_filenames, compression_type='')
train_dataset_iter = train_dataset.as_numpy_iterator()

val_filenames = tf.io.matching_files(VAL_FILES)
val_dataset = tf.data.TFRecordDataset(val_filenames, compression_type='')
val_dataset_iter = val_dataset.as_numpy_iterator()

test_filenames = tf.io.matching_files(TEST_FILES)
test_dataset = tf.data.TFRecordDataset(test_filenames, compression_type='')
test_dataset_iter = test_dataset.as_numpy_iterator()

CAMERA_NAME_ENUM = {
    2: "FRONT_LEFT",
    1: "FRONT",
    3: "FRONT_RIGHT"
}
intent_map = { "UNKNOWN": 0,
              "GO_STRAIGHT": 1,
               "GO_LEFT": 2,
               "GO_RIGHT": 3
}

def parse_e2ed_frame_with_past_dynamics_and_front3(bytes_example, image_size=(224, 224)):
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(bytes_example)

    # --- Extract front 3 images and calibrations ---
    image_list = []
    order = [2, 1, 3]
    for camera_name in order:  # FRONT_LEFT, FRONT, FRONT_RIGHT
        found = False
        for idx, img_content in enumerate(data.frame.images):
            if img_content.name == camera_name :
                image = tf.io.decode_jpeg(img_content.image)
                image = tf.image.resize(image, image_size)
                image = tf.cast(image, tf.float32) / 255.0
                image_list.append(image)
                found = True
                break
        if not found:
            return None  # Skip this sample if any of the 3 images are missing

    image_stack = tf.stack(image_list, axis=0)  # [3, H, W, 3]

    # --- Past dynamics ---
    past_dyn = np.stack([
        data.past_states.pos_x,
        data.past_states.pos_y,
        data.past_states.vel_x,
        data.past_states.vel_y,
        data.past_states.accel_x,
        data.past_states.accel_y], axis=1)  #[16.3]


    # --- Intent ---
    intent_label = data.intent
    intent_one_hot = tf.one_hot(intent_label, depth=4)

    # --- Future (x, y) trajectory ---
    future_xy = np.stack([
        data.future_states.pos_x,
        data.future_states.pos_y,
        data.future_states.pos_z
    ], axis=1) # [20,3]

    return ((image_stack, intent_one_hot, past_dyn), future_xy)

def load_n_samples_from_iterator(dataset_iter, N, image_size=(224, 224)):
    X_images, X_intents, X_past_dyn, Y_future_xy = [], [], [], []

    count = 0
    while count < N:
        try:
            bytes_example = next(dataset_iter)
            result = parse_e2ed_frame_with_past_dynamics_and_front3(bytes_example, image_size=image_size)
            if result is None:
                continue  # skip incomplete frames
            (images, intent, past_dyn), future_xy = result

            X_images.append(np.transpose(images,(0,3,1,2)))        # shape: [3, 224, 224, 3]
            X_intents.append(intent)       # shape: [4]
            X_past_dyn.append(past_dyn)    # shape: [16, 6]
            Y_future_xy.append(future_xy)  # shape: [20, 3]

            count += 1
        except StopIteration:
            break

    return (
        np.stack(X_images),        # [N, 3, 224, 224, 3]
        np.stack(X_intents),       # [N, 4]
        np.stack(X_past_dyn),      # [N, 16, 6]
        np.stack(Y_future_xy),     # [N, 20, 3]
    )


class WaymoTrajectoryDataset(Dataset):
    def __init__(self, dataset_iter, n_samples=10):
        image_tensor, intent_tensor, past_dyn_tensor, target_tensor = load_n_samples_from_iterator(dataset_iter, N=n_samples)
        self.images = torch.from_numpy(image_tensor).float()         # [N, 3, 224, 224, 3]
        self.intents = torch.from_numpy(intent_tensor).float()       # [N, 4]
        self.past_dyn = torch.from_numpy(past_dyn_tensor).float()    # [N, 16, 6]
        self.targets = torch.from_numpy(target_tensor).float()       # [N, 20, 3]

        transform_nn = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
        ])

        self.transform = lambda x: transform_nn(x)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        processed_images = []
        
        for img in self.images[idx]:
            processed_images.append(self.transform(img))
        self.images[idx] = torch.stack(processed_images)
        
        return {
            "images": self.images[idx],       # [3, 3, 224, 224]
            "intent": self.intents[idx],      # [4]
            "past_dyn": self.past_dyn[idx],   # [16, 6]
            "target": self.targets[idx]       # [20, 3]
        }

# # # Wrap in PyTorch Dataset
# train_dataset = WaymoTrajectoryDataset(train_dataset_iter)
# val_dataset = WaymoTrajectoryDataset(val_dataset_iter, n_samples=10)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
# # Test a batch
# for batch in train_loader:
#     print(batch["images"].shape)    # [8, 3, 3, 224, 224]
#     print(batch["intent"].shape)    # [8, 4]
#     print(batch["past_dyn"].shape)  # [8, 16, 6]
#     print(batch["target"].shape)    # [8, 20, 3]
#     break