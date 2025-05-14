import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from torchvision import datasets
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os
import matplotlib.pyplot as plt

class FaceEmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.samples = []
        
        
        # Build dataset
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def extract_face(self, image_path):
        try:
            # Detect faces using DeepFace
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                target_size=(224, 224),
                detector_backend='opencv',  # Use OpenCV for faster detection
                enforce_detection=False  # Don't raise error if face not detected
            )
            
            if face_objs and len(face_objs) > 0:
                # Get the first face
                face = face_objs[0]['face']
                # Convert numpy array to PIL Image
                face = Image.fromarray(face)
                return face
            
        except Exception as e:
            return None
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Extract face
        face = self.extract_face(img_path)
        
        if face is None:
            # If no face detected, return the original image
            face = Image.open(img_path).convert('RGB')
        
        if self.transform:
            face = self.transform(face)
            
        return face, label

    def get_labels(self):
        return [sample[1] for sample in self.samples]

# Image preprocessing
image_size = 224  # Changed to standard size for ResNet

transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_set = FaceEmotionDataset(
    root_dir="Tu_lam_v2_11/train",
    transform=transform_train
)

test_set = FaceEmotionDataset(
    root_dir="Tu_lam_v2_11/test",
    transform=transform_test
)

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores

# Initialize model
convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.Flatten()
)

model = PrototypicalNetworks(convolutional_network)
if torch.cuda.is_available():
    model = model.cuda()

# Few-shot parameters
N_WAY = 5  # Number of emotions per task
N_SHOT = 5  # Number of examples per emotion
N_QUERY = 10  # Number of query images per emotion
N_EVALUATION_TASKS = 100

from easyfsl.samplers import TaskSampler

test_sampler = TaskSampler(
    test_set, 
    n_way=N_WAY, 
    n_shot=N_SHOT, 
    n_query=N_QUERY, 
    n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    
    if torch.cuda.is_available():
        support_images = support_images.cuda()
        support_labels = support_labels.cuda()
        query_images = query_images.cuda()
        query_labels = query_labels.cuda()
    
    scores = model(support_images, support_labels, query_images)
    predictions = torch.max(scores.detach().data, 1)[1]
    return (predictions == query_labels).sum().item(), len(query_labels)

def evaluate(data_loader: DataLoader):
    total_predictions = 0
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):

            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels
            )

            total_predictions += total
            correct_predictions += correct

    accuracy = 100 * correct_predictions/total_predictions
    print(f"Model tested on {len(data_loader)} tasks. Accuracy: {accuracy:.2f}%")
    return accuracy

# Training setup
N_TRAINING_EPISODES = 40000
N_VALIDATION_TASKS = 100

train_sampler = TaskSampler(
    train_set, 
    n_way=N_WAY, 
    n_shot=N_SHOT, 
    n_query=N_QUERY, 
    n_tasks=N_TRAINING_EPISODES
)

train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_episode(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    
    if torch.cuda.is_available():
        support_images = support_images.cuda()
        support_labels = support_labels.cuda()
        query_images = query_images.cuda()
        query_labels = query_labels.cuda()
    
    classification_scores = model(support_images, support_labels, query_images)
    loss = criterion(classification_scores, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def train():
    best_accuracy = 0
    log_update_frequency = 10
    all_loss = []
    
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) in tqdm_train:
            loss_value = train_episode(support_images, support_labels, query_images, query_labels)
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                avg_loss = sum(all_loss[-log_update_frequency:]) / log_update_frequency
                tqdm_train.set_postfix(loss=avg_loss)
            
            # Evaluate every 200 episodes
            if episode_index > 0 and episode_index % 200 == 0:
                accuracy = evaluate(test_loader)
                accuracy_history.append(accuracy)
                episode_history.append(episode_index)
                model.train()
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), 'best_emotion_model.pth')
                
                # Plot accuracy over time
                plt.figure(figsize=(10, 5))
                plt.plot(episode_history, accuracy_history, 'b-')
                plt.title('Accuracy over Training Episodes')
                plt.xlabel('Episode')
                plt.ylabel('Accuracy (%)')
                plt.grid(True)
                plt.savefig('accuracy_progress.png')
                plt.close()

if __name__ == "__main__":
    print("Starting training...")
    train()
    print("Training completed!")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_emotion_model.pth'))
    print("\nFinal evaluation on test set:")
    evaluate(test_loader) 