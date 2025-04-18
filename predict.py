import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List, Dict
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import models
# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Tasks and corresponding domain folders
domain_tasks = {
    "bcn_vs_ham_age_u30": "bcn_age_u30",
    "msk_vs_bcn_age_u30": "msk_age_u30",
    "bcn_vs_ham_loc_head_neck": "bcn_loc_head_neck",
    "bcn_vs_msk_headloc": "bcn_loc_head_neck",
    "ham_vs_msk_loc_head_neck": "ham_loc_head_neck",
    "ham_age_u30vsmsk_age_u30": "ham_age_u30",
    "bcn_vs_ham_loc_palms_soles": "bcn_loc_palms_soles"
}

class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        # print("This is shape of input:",x.shape)
        # if(x.ndim == 3):
        #     x = x.unsqueeze(0)
        f = self.backbone(x)
       
        if(f.ndim == 2):
            f = f.unsqueeze(0)
        # print("This is shape of features:",f.shape)
        f = self.pool_layer(f)
        

        # print("Shape before bottleneck:", f.shape)  # Should be (batch_size, 2048)
       

        f = self.bottleneck(f)
        # print("Shape after bottleneck:", f.shape)   
        predictions = self.head(f)

        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        # Determine the number of output features from the backbone
        if hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Linear):
            out_features = backbone.fc.in_features
        elif hasattr(backbone, 'fc'):
            # Try to infer the number of input features
            dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size according to model input
            backbone.eval()
            with torch.no_grad():
                features = backbone(dummy_input)
            out_features = features.shape[1]  # Extract the feature dimension
        else:
            raise ValueError("The backbone does not have a fc layer.")
        
        bottleneck = nn.Sequential(
            nn.Linear(out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
      
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits, features  # Return both logits and features



def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_matching_model(num_classes=2, bottleneck_dim=512):
    backbone = get_model("resnet18", pretrain=False)
    backbone.fc = nn.Identity()

    model = ImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        bottleneck_dim=bottleneck_dim,
    )
    return model

def compute_feature(model, image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.backbone(x)
        feat = torch.flatten(feat, 1)
        pred = model.head(model.bottleneck(feat))
        predicted_label = torch.argmax(pred, dim=1).item()

    feat = feat.squeeze().cpu().numpy()
    feat /= np.linalg.norm(feat)
    return feat, predicted_label

def average_adistance(image_feat, features_path):
    domain_features = np.load(features_path)
    domain_features = domain_features / np.linalg.norm(domain_features, axis=1, keepdims=True)
    distances = np.linalg.norm(domain_features - image_feat, axis=1)
    return np.mean(distances)

def predict_task_label(image_path, model_dir, feat_dir):
    min_distance = float("inf")
    best_task = None
    best_label = None

    for task, domain in domain_tasks.items():
        print(f"🔍 Evaluating task: {task}")

        # Load model
        model = get_matching_model()
        pth_path = os.path.join(model_dir, f"{task}.log", "checkpoints", "best.pth").replace("\\", "/")

        model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)
        model.to(device)
        model.eval()

        # Extract feature from image
        image_feat, label = compute_feature(model, image_path)

        # Load features and compute A-distance
        features_path = os.path.join(feat_dir, f"{task}_features.npy")
        if not os.path.exists(features_path):
            print(f"⚠️ Skipping {task}, feature file missing.")
            continue

        adist = average_adistance(image_feat, features_path)
        print(f"   → Avg A-distance: {adist:.4f}")

        if adist < min_distance:
            min_distance = adist
            best_task = task
            best_label = label

    return best_task, min_distance, best_label



if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    # image_path = "C:/Users/sanji/Documents/NewMedScan/uploads/1744809643583.png"
    model_dir = "C:\\Users\\sanji\\Documents\\NewMedScan\\model_checkpoints"

    feat_dir = "./domain_features_dann"
    task, distance, label = predict_task_label(image_path, model_dir, feat_dir)
    print(task)
    print(distance)
    print(label)
