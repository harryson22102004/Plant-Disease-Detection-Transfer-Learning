import torch, torch.nn as nn
import torchvision.models as models, torchvision.transforms as T
 
DISEASES=['healthy','leaf_blight','leaf_rust','powdery_mildew','bacterial_spot']
 
class PlantDiseaseModel(nn.Module):
    def __init__(self, n_classes=5, backbone='resnet50'):
        super().__init__()
        bb=models.resnet50(pretrained=False)
        # Freeze early layers (transfer learning)
        for p in list(bb.parameters())[:-20]: p.requires_grad=False
        bb.fc=nn.Sequential(nn.Dropout(0.5),nn.Linear(2048,512),nn.ReLU(),nn.Linear(512,n_classes))
        self.model=bb
    def forward(self,x): return self.model(x)
 
def get_transforms(train=True):
    if train:
        return T.Compose([T.Resize(256),T.RandomCrop(224),T.RandomHorizontalFlip(),
                           T.ColorJitter(0.2,0.2,0.2,0.1),T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
 
def focal_loss(logits, targets, gamma=2.0):
    ce=nn.CrossEntropyLoss(reduction='none')(logits,targets)
    pt=torch.exp(-ce)
    return ((1-pt)**gamma*ce).mean()
 
model=PlantDiseaseModel(5)
x=torch.randn(4,3,224,224); y=torch.randint(0,5,(4,))
out=model(x); loss=focal_loss(out,y)
print(f"Output: {out.shape} | Focal loss: {loss.item():.3f}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Classes: {DISEASES}")
