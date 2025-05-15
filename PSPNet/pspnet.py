import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# Label definitions (same as original)
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

# PSPNet Implementation
class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        for size in sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True)
                )
            )
        self.stages = nn.ModuleList(self.stages)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + (in_channels // 4 * len(sizes)), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        feats = [x]
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            feats.append(feat)
        out = torch.cat(feats, dim=1)
        return self.project(out)

class PSPNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(PSPNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.psp = PSPModule(in_channels=2048)
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.psp(x)
        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return {'out': x}

class IDDDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.masks = []
        img_dir = os.path.join(root, 'leftImg8bit', split)
        mask_dir = os.path.join(root, 'gtFine', split)
        cities = os.listdir(img_dir)
        for city in cities:
            img_files = os.listdir(os.path.join(img_dir, city))
            for img_file in img_files:
                if not img_file.endswith('.png'):
                    continue
                img_id = img_file.split('_')[0]
                img_path = os.path.join(img_dir, city, img_file)
                mask_name = img_id + '_gtFine_labelids.png'
                mask_path = os.path.join(mask_dir, city, mask_name)
                if not os.path.exists(mask_path):
                    continue
                self.images.append(img_path)
                self.masks.append(mask_path)
        assert len(self.images) > 0, "No images found in dataset!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
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
        return image, mask_mapped

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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc='Training', leave=False)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_list = []
    loop = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            ious = compute_iou(preds.cpu(), masks.cpu(), num_classes, ignore_label)
            iou_list.append(ious)
            loop.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_iou = np.nanmean(np.array(iou_list), axis=0)
    return epoch_loss, mean_iou

def main():
    dataset_root = "IDD_Segmentation"  # Update to your dataset path
    batch_size = 8
    num_epochs = 30
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Ensure consistent size
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    target_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),  # Resize masks with nearest neighbor
        transforms.Lambda(lambda pic: torch.from_numpy(np.array(pic)).long())
    ])
    train_dataset = IDDDataset(dataset_root, split='train', transform=transform, target_transform=target_transform)
    val_dataset = IDDDataset(dataset_root, split='val', transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    model = PSPNet(num_classes=num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean IoU: {np.nanmean(val_iou):.4f}")
    torch.save(model.state_dict(), "pspnet_idd.pth")

if __name__ == "__main__":
    main()