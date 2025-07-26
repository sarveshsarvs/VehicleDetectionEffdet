import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from effdet import create_model
import torch.nn as nn
import torch.nn.functional as F

#load dataset from coco
class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        #load the annotations from json
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        #extract image file paths and annotations
        self.image_files = [os.path.join(image_dir, img['file_name']) for img in self.data['images']]
        self.annotations = self.data['annotations']

        #number of annotations loaded for debug
        print(f"Loaded {len(self.annotations)} annotations.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        

        #resize the image to 512x512
        image = image.resize((512, 512))
        
        if self.transform:
            image = self.transform(image)
        
        #load annotations for the current image
        target = [ann for ann in self.annotations if ann['image_id'] == self.data['images'][idx]['id']]
        
        #print warning if no annotations are found for image
        if len(target) == 0:
            print(f"Warning: No annotations for image index {idx} with image_id {self.data['images'][idx]['id']}")
        
        #convert annotations to tensors (the pain this mf inflicted on me)
        target = {
            'boxes': torch.tensor([ann['bbox'] for ann in target], dtype=torch.float32),
            'labels': torch.tensor([ann['category_id'] for ann in target], dtype=torch.int64),
        }
        
        return image, target

#custom collate function to handle batching (this guy... i dont want to talk about this)
def collate_fn(batch):
    images, targets = zip(*batch)
    
    #stack images into a 4D tensor (batch_size, channels, height, width)
    images = torch.stack([img for img in images], dim=0)
    
    #targets need to be handled as a list of dictionaries
    targets = [{k: v for k, v in t.items()} for t in targets]
    
    return images, targets

#data transformations for preprocessing images (the previous resize wasnt enough smh)
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

#initialize dataset and dataloaders
train_dataset = CustomDataset(
    image_dir='/train/images/directory/path/',
    annotation_file='/annotations/path/annotations_coco.json',
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=7,  #set batch size 8
    shuffle=True,
    num_workers=4,  #number of cores for data loading
    collate_fn=collate_fn  #use custom collate function (as if mf did it default)
)

#focal Loss implementation (oh boy oh boy where do i even start with the issues i had with crossentropyloss)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        #compute cross-entropy loss (and using this indirectly has no issues??? Blasphemy!)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        #calculate focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

#filter predictions to match targets (this guy is the GOAT, else i end up with a million predictions[40000~])
def filter_predictions(predictions, targets):
    class_preds, bbox_preds = predictions

    filtered_class_preds = []
    filtered_bbox_preds = []
    filtered_targets = []

    for class_pred, bbox_pred, target in zip(class_preds, bbox_preds, targets):
        # Flatten predictions if they are 4D tensors
        if class_pred.dim() == 4:
            class_pred_flat = class_pred.view(class_pred.size(0), -1, class_pred.size(1)).permute(0, 2, 1).contiguous().view(-1, class_pred.size(1))
        else:
            class_pred_flat = class_pred.view(-1, class_pred.size(-1))

        if bbox_pred.dim() == 4:
            bbox_pred_flat = bbox_pred.view(bbox_pred.size(0), -1, 4)
        else:
            bbox_pred_flat = bbox_pred.view(-1, 4)

        # Extract targets
        target_boxes = target['boxes']
        target_labels = target['labels']

        num_targets = target_labels.size(0)
        num_preds = class_pred_flat.size(0)

        if num_targets > num_preds:
            raise ValueError(f"Number of targets ({num_targets}) is greater than number of predictions ({num_preds})")

        # Filter class predictions and bounding boxes
        filtered_class_pred = class_pred_flat[:num_targets]
        filtered_bbox_pred = bbox_pred_flat[:num_targets]

        filtered_class_preds.append(filtered_class_pred)
        filtered_bbox_preds.append(filtered_bbox_pred)
        filtered_targets.append({
            'boxes': target_boxes[:num_targets],
            'labels': target_labels[:num_targets]
        })

    return (filtered_class_preds, filtered_bbox_preds), filtered_targets


#custom loss function (i hate this shit with every cell of my body, no amount of therapy wil suffice for the trauma this bought upon me)
class EfficientDetLoss(nn.Module):
    def __init__(self, num_classes):
        super(EfficientDetLoss, self).__init__()
        self.num_classes = num_classes
        self.classification_loss = FocalLoss()
        self.bbox_regression_loss = nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        class_preds, bbox_preds = predictions

        # Flatten predictions
        class_preds_flat = []
        bbox_preds_flat = []

        for level_class_pred, level_bbox_pred in zip(class_preds, bbox_preds):
            if level_class_pred.dim() == 4:
                class_preds_flat.append(level_class_pred.view(level_class_pred.size(0), -1, level_class_pred.size(1)).permute(0, 2, 1).contiguous().view(-1, level_class_pred.size(1)))
            else:
                class_preds_flat.append(level_class_pred.view(-1, level_class_pred.size(-1)))

            if level_bbox_pred.dim() == 4:
                bbox_preds_flat.append(level_bbox_pred.view(level_bbox_pred.size(0), -1, 4))
            else:
                bbox_preds_flat.append(level_bbox_pred.view(-1, 4))

        # Filter predictions and targets
        (filtered_class_preds, filtered_bbox_preds), filtered_targets = filter_predictions((class_preds_flat, bbox_preds_flat), targets)

        total_class_loss = 0
        total_bbox_loss = 0

        for i, target in enumerate(filtered_targets):
            class_targets = target['labels']
            bbox_targets = target['boxes']

            # Flatten targets
            class_targets_flat = class_targets.view(-1)
            bbox_targets_flat = bbox_targets.view(-1, 4)

            num_preds = filtered_class_preds[i].size(0)
            num_targets = class_targets_flat.size(0)

            if num_targets > num_preds:
                raise ValueError(f"Number of targets ({num_targets}) is greater than number of predictions ({num_preds})")

            # Adjust targets to match number of predictions
            if num_preds > num_targets:
                num_repeats = num_preds // num_targets
                class_targets_flat = class_targets_flat.repeat(num_repeats)

                if class_targets_flat.size(0) != num_preds:
                    raise ValueError(f"Number of targets ({class_targets_flat.size(0)}) does not match number of predictions ({num_preds})")

            assert class_targets_flat.size(0) == filtered_class_preds[i].size(0), "Target size does not match prediction size."

            # Calculate loss
            class_loss = self.classification_loss(filtered_class_preds[i], class_targets_flat)
            total_class_loss += class_loss

            bbox_loss = self.bbox_regression_loss(filtered_bbox_preds[i], bbox_targets_flat)
            total_bbox_loss += bbox_loss

        # Compute total loss as the sum of classification and bounding box losses
        total_loss = (total_class_loss / len(filtered_targets)) + (total_bbox_loss / len(filtered_targets))

        return total_loss


#initialize model
model = create_model('tf_efficientdet_lite3', pretrained=True, num_classes=6)  #(car, bike, rickshaw, cark, truck, ambulance)
num_classes = 6
loss_fn = EfficientDetLoss(num_classes)

#training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 2000

#initialize variables for saving the best model
best_loss = float('inf')  #set an initial high value 
model_save_path = 'efficientdet.pth'

#training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = images.to(device)  #move images to the device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()  #clear previous gradients
        
        #forward pass
        predictions = model(images)
        
        #calculate losses
        total_loss = loss_fn(predictions, targets)
        
        total_loss.backward()  #compute gradients
        optimizer.step()  #update weights
        
        epoch_loss += total_loss.item()  #accumulate loss for the epoch

    #calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss}")

    #save the best model based on the lowest loss
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved with loss: {best_loss}")

print("Training complete!")
