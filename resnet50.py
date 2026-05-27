import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import warnings
import random
from tqdm import tqdm
import time
import sys

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import label_binarize

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "/nfsshare/users/raghavan/sensing and imaging/SI/"
OUTPUT_PATH = "/nfsshare/users/raghavan/sensing and imaging/SI/output12/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Checking dataset folders...")
if os.path.exists(DATASET_PATH):
    all_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
   
    CLASS_NAMES = []
    for folder in all_folders:
        if folder.startswith('.') or folder.startswith('_'):
            continue
        folder_path = os.path.join(DATASET_PATH, folder)
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)])
        if image_count > 0:
            CLASS_NAMES.append(folder)
   
    print(f"Classes: {CLASS_NAMES}")
else:
    print(f"ERROR: Path {DATASET_PATH} does not exist!")
    sys.exit(1)

NUM_CLASSES = len(CLASS_NAMES)

TARGET_PER_CLASS = 110
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
PATIENCE = 10
WEIGHT_DECAY = 0.0001

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
class RetinalImageLoader:
    def __init__(self, root_path, class_names):
        self.root_path = root_path
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.image_paths = []
        self.labels = []

    def load_images(self):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        for class_name in self.class_names:
            class_path = os.path.join(self.root_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(image_extensions)]
            class_idx = self.class_to_idx[class_name]
            for img_file in images:
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(class_idx)
            print(f" {class_name}: {len(images)} images")
        return self.image_paths, np.array(self.labels)

loader = RetinalImageLoader(DATASET_PATH, CLASS_NAMES)
all_paths, all_labels = loader.load_images()

# ============================================================
# STEP 2: STRATIFIED SPLIT
# ============================================================
def stratified_split(paths, labels, test_size=0.1, val_size=0.2):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(sss1.split(paths, labels))
   
    train_val_paths = [paths[i] for i in train_val_idx]
    train_val_labels = labels[train_val_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = labels[test_idx]
   
    val_ratio = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss2.split(train_val_paths, train_val_labels))
   
    train_paths = [train_val_paths[i] for i in train_idx]
    train_labels = train_val_labels[train_idx]
    val_paths = [train_val_paths[i] for i in val_idx]
    val_labels = train_val_labels[val_idx]
   
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
    stratified_split(all_paths, all_labels)

# ============================================================
# STEP 3: DATA AUGMENTATION (FIXED)
# ============================================================
class DataAugmenter:
    def __init__(self, target_per_class, output_dir):
        self.target_per_class = target_per_class
        self.augmented_dir = os.path.join(output_dir, 'augmented_temp')
        os.makedirs(self.augmented_dir, exist_ok=True)

    def augment_image(self, img, augment_type='heavy'):
        h, w = img.shape[:2]
        if augment_type == 'heavy':
            angle = np.random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
            scale = np.random.uniform(0.8, 1.2)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            if new_w > w or new_h > h:
                img = img[:h, :w]
            else:
                pad_w = (w - new_w) // 2
                pad_h = (h - new_h) // 2
                img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT, value=0)
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            contrast = np.random.uniform(0.8, 1.2)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        else:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
            brightness = np.random.uniform(0.85, 1.15)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        return img

    def balance_class(self, image_paths, labels, class_idx):
        class_indices = [i for i, lbl in enumerate(labels) if lbl == class_idx]
        class_paths = [image_paths[i] for i in class_indices]
        current_count = len(class_paths)
       
        if current_count == 0 or current_count >= self.target_per_class:
            return image_paths, labels
       
        needed = self.target_per_class - current_count
        class_name = CLASS_NAMES[class_idx]
        augment_type = 'heavy' if current_count < 30 else 'light'
       
        print(f" {class_name}: {current_count} -> {self.target_per_class} (+{needed})")
       
        augmented_paths = []
        augmented_labels = []
       
        for i in range(needed):
            orig_idx = np.random.randint(0, current_count)
            img_path = class_paths[orig_idx]
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_aug = self.augment_image(img, augment_type)
           
            aug_filename = f"{class_name}_aug_{orig_idx}_{i}.jpg"
            aug_path = os.path.join(self.augmented_dir, class_name, aug_filename)
            os.makedirs(os.path.dirname(aug_path), exist_ok=True)
            Image.fromarray(img_aug).save(aug_path)
           
            augmented_paths.append(aug_path)
            augmented_labels.append(class_idx)
       
        return image_paths + augmented_paths, labels + augmented_labels

    def balance_dataset(self, image_paths, labels):
        balanced_paths = list(image_paths)
        balanced_labels = list(labels)
        
        for class_idx in range(NUM_CLASSES):
            balanced_paths, balanced_labels = self.balance_class(
                balanced_paths, balanced_labels, class_idx)
        return balanced_paths, balanced_labels


augmenter = DataAugmenter(TARGET_PER_CLASS, OUTPUT_PATH)
train_paths_augmented, train_labels_augmented = augmenter.balance_dataset(train_paths, train_labels)

print(f"\nTotal training samples after augmentation: {len(train_paths_augmented)}")

# ============================================================
# STEP 4-6: DATASET & DATALOADERS
# ============================================================
class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
            except:
                image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = RetinalDataset(train_paths_augmented, train_labels_augmented, train_transform)
val_dataset = RetinalDataset(val_paths, val_labels, val_test_transform)
test_dataset = RetinalDataset(test_paths, test_labels, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# STEP 7: CLASS WEIGHTS
# ============================================================
original_train_counts = [np.sum(np.array(train_labels) == i) for i in range(NUM_CLASSES)]
total_original = sum(original_train_counts)
class_weights_list = [total_original / (NUM_CLASSES * c) if c > 0 else 1.0 for c in original_train_counts]
class_weights = torch.FloatTensor(class_weights_list).to(device)

# ============================================================
# STEP 8: MODEL
# ============================================================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(device)

# ============================================================
# STEP 9: TRAINING SETUP (FIXED)
# ============================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# FIXED: Removed 'verbose=True'
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

print("Training setup completed (scheduler fixed)")

# ============================================================
# TRAINING LOOP (STEP 10)
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

# Training
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0
patience_counter = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'best_model.pth'))
        patience_counter = 0
        print(f"*** Best model saved! Val Acc: {val_acc:.2f}% ***")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered!")
            break

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

print(f"\nTraining completed in {(time.time()-start_time)/60:.1f} minutes")
