import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

# Label definitions
labels = [
    ('road', 0, False, (128, 64, 128)),
    ('parking', 1, False, (250, 170, 160)),
    ('drivable fallback', 2, False, (81, 0, 81)),
    ('sidewalk', 3, False, (244, 35, 232)),
    ('rail track', 4, False, (230, 150, 140)),
    ('non-drivable fallback', 5, False, (152, 251, 152)),
    ('person', 6, False, (220, 20, 60)),
    ('animal', 7, True, (246, 198, 145)),
    ('rider', 8, False, (255, 0, 0)),
    ('motorcycle', 9, False, (0, 0, 230)),
    ('bicycle', 10, False, (119, 11, 32)),
    ('autorickshaw', 11, False, (255, 204, 54)),
    ('car', 12, False, (0, 0, 142)),
    ('truck', 13, False, (0, 0, 70)),
    ('bus', 14, False, (0, 60, 100)),
    ('caravan', 15, True, (0, 0, 90)),
    ('trailer', 16, True, (0, 0, 110)),
    ('train', 17, True, (0, 80, 100)),
    ('vehicle fallback', 18, False, (136, 143, 153)),
    ('curb', 19, False, (220, 190, 40)),
    ('wall', 20, False, (102, 102, 156)),
    ('fence', 21, False, (190, 153, 153)),
    ('guard rail', 22, False, (180, 165, 180)),
    ('billboard', 23, False, (174, 64, 67)),
    ('traffic sign', 24, False, (220, 220, 0)),
    ('traffic light', 25, False, (250, 170, 30)),
    ('pole', 26, False, (153, 153, 153)),
    ('polegroup', 27, False, (153, 153, 153)),
    ('obs-str-bar-fallback', 28, False, (169, 187, 214)),
    ('building', 29, False, (70, 70, 70)),
    ('bridge', 30, False, (150, 100, 100)),
    ('tunnel', 31, False, (150, 120, 90)),
    ('vegetation', 32, False, (107, 142, 35)),
    ('sky', 33, False, (70, 130, 180)),
    ('fallback background', 34, False, (169, 187, 214)),
    ('unlabeled', 35, True, (0, 0, 0)),
    ('ego vehicle', 36, True, (0, 0, 0)),
    ('rectification border', 37, True, (0, 0, 0)),
    ('out of roi', 38, True, (0, 0, 0)),
    ('license plate', 39, True, (0, 0, 142))
]

ignore_label = 255
num_classes = 19

id_to_trainid = {}
trainid_to_name = {}
trainid = 0
for (name, label_id, ignore, color) in labels:
    if ignore:
        id_to_trainid[label_id] = ignore_label
    else:
        id_to_trainid[label_id] = trainid
        trainid_to_name[trainid] = name
        trainid += 1
        if trainid >= num_classes:
            break

palette = []
for _, _, _, color in labels[:num_classes]:
    palette.extend(color)
palette.extend([0] * (768 - len(palette)))

def colorize_mask(mask):
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    mask_img.putpalette(palette)
    return mask_img

class IDDDataset(Dataset):
    def __init__(self, root, split='val', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.masks = []
        self.filenames = []  # Store original filenames for output
        img_dir = os.path.join(root, 'leftImg8bit', split)
        mask_dir = os.path.join(root, 'gtFine', split)

        # Debugging: Check if directories exist
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory {img_dir} does not exist.")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory {mask_dir} does not exist.")

        cities = os.listdir(img_dir) if os.path.exists(img_dir) else []
        if not cities:
            raise ValueError(f"No city subdirectories found in {img_dir}.")

        for city in cities:
            city_img_dir = os.path.join(img_dir, city)
            if not os.path.isdir(city_img_dir):
                print(f"Skipping non-directory {city_img_dir}")
                continue
            img_files = os.listdir(city_img_dir)
            if not img_files:
                print(f"Warning: No files found in {city_img_dir}")
                continue
            for img_file in img_files:
                if not img_file.endswith('.png'):
                    continue
                img_id = img_file.split('_')[0]
                img_path = os.path.join(img_dir, city, img_file)
                mask_name = img_id + '_gtFine_labelids.png'
                mask_path = os.path.join(mask_dir, city, mask_name)
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask {mask_path} not found for image {img_path}")
                    continue
                self.images.append(img_path)
                self.masks.append(mask_path)
                self.filenames.append(img_file)

        if len(self.images) == 0:
            raise ValueError(f"No valid image-mask pairs found in {img_dir} for split: {split}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        filename = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_mapped = np.ones_like(mask_np) * ignore_label
        for k, v in id_to_trainid.items():
            mask_mapped[mask_np == k] = v
        mask_mapped = Image.fromarray(mask_mapped.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask_mapped = self.target_transform(mask_mapped)
        return image, mask_mapped, filename

def compute_iou(pred, target, num_classes, ignore_index=255):
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.array(ious)

def test(model, dataloader, device, output_dir="deeplabv3_outputs"):
    model.eval()
    iou_list = []
    os.makedirs(output_dir, exist_ok=True)
    loop = tqdm(dataloader, desc='Testing', leave=False)
    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            ious = compute_iou(preds.cpu(), masks.cpu(), num_classes, ignore_label)
            iou_list.append(ious)
            # Save all predicted masks
            for i, (pred, filename) in enumerate(zip(preds, filenames)):
                pred_np = pred.cpu().numpy()
                pred_colored = colorize_mask(pred_np)
                output_filename = f"pred_{os.path.splitext(filename)[0]}.png"
                pred_colored.save(os.path.join(output_dir, output_filename))
    mean_iou = np.nanmean(np.array(iou_list), axis=0)
    return mean_iou

def main():
    dataset_root = "IDD_Segmentation"  # Update to your dataset path
    model_path = "deeplabv3_idd.pth"  # Path to trained model weights
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    target_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic)).long())
    ])

    # Load validation dataset as test set
    try:
        test_dataset = IDDDataset(dataset_root, split='val', transform=transform, target_transform=target_transform)
        if len(test_dataset) == 0:
            raise ValueError("Validation dataset is empty")
    except Exception as e:
        raise ValueError(f"Failed to load validation split: {e}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    model = deeplabv3_resnet50(num_classes=num_classes)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Starting testing phase on validation split...")
    test_iou = test(model, test_loader, device)
    mIoU = np.nanmean(test_iou) * 100  # Convert to percentage
    print(f"Test mIoU: {mIoU:.2f}%")
    for i, iou in enumerate(test_iou):
        if not np.isnan(iou):
            print(f"Class {trainid_to_name[i]} IoU: {iou*100:.2f}%")
    print(f"Saved {len(test_dataset)} predicted masks in deeplabv3_outputs")

if __name__ == "__main__":
    main()