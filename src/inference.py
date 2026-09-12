import torch 
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from ultralytics import YOLO
import os


WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH",'./weights')
SAMPLE_PATH = os.environ.get("SAMPLE_PATH",'./sample_data')

cls_checkpoint = torch.load(os.path.join(WEIGHTS_PATH,'cls.pth'))

det_model = YOLO(os.path.join(WEIGHTS_PATH,'objectdetection.pt'))
cls_model = efficientnet_b3()


cls_trasnform = transforms.Compose([transforms.Resize((300,300))
                                      ,transforms.ToTensor()
                                      ,transforms.Normalize(
                                            (0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225)
                                      )])

normal_transform = transforms.Compose([transforms.Resize((512,512))
                                     ,transforms.ToTensor()])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cls_model.classifier[1] = nn.Linear(cls_model.classifier[1].in_features,4)
cls_labels = ['glioma','meningioma','normal','pituitary']

cls_model.load_state_dict(cls_checkpoint['model_state_dict'])
cls_model.to(device)

target_layers = [cls_model.features[-1][0]]
cam = GradCAM(cls_model,target_layers)

def tumorpredict(path,det_model,cls,cls_transform,normal_transform):
    cls.eval()
    names = ['gl','me','nt','pi']
    id = 0
    for idx,name in enumerate(names):
        if (name in path):
            id = idx
            break

    PILimg = Image.open(path).convert("RGB")
    det_res = det_model.predict(path,conf=0.5)
    cls_img = cls_transform(PILimg)
    image = normal_transform(PILimg)
    cls_img = cls_img.to('cuda')
    cls_img = cls_img.unsqueeze(0)
    cls_pred = cls(cls_img)
    cls_pred = cls_pred.argmax(dim=1)
    target = [ClassifierOutputTarget(int(cls_pred.item()))]
    #cam(expects(N,C,H,W),expects 1D array[])
    graycam = cam(cls_img,target)[0]
    #graycam returns (1,H,W)
    fig,axes = plt.subplots(1,3)
    #cam_on_image(unnormalized numpy array in (H,W,C) , (H,W))
    cls_img_display = cls_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    cls_img_display = cls_img_display*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406])
    cls_img_display = np.clip(cls_img_display,0,1)
    visualization = show_cam_on_image(cls_img_display,graycam,use_rgb=True)
    axes[0].imshow(image.permute(1,2,0).cpu().numpy())
    axes[0].set_xlabel(f"Actual : {cls_labels[id]}")
    axes[0].set_ylabel(f"Predicted : {cls_labels[cls_pred.item()]}")
    axes[0].set_title("Input")
    axes[1].imshow(visualization)
    axes[1].set_title("Grad-Cam")
    axes[2].imshow(image.permute(1,2,0).cpu().numpy())
    box_img = det_res[0].plot()
    axes[2].imshow(box_img[:,:,::-1])
    axes[2].set_title("Object Detection")
    plt.tight_layout()
    plt.show()
    plt.close()
    return (det_res[0].boxes.xyxy)

def predictarea(box):
    if len(box)>0:
        width = box[0][2]-box[0][0]
        height = box[0][3]-box[0][1]
        return width*height
    return None


box=tumorpredict(os.path.join(SAMPLE_PATH,'gl1.jpg'),det_model,cls_model,cls_trasnform,normal_transform)
aprox_area = predictarea(box)



#Just another Side Feature not a Main Feature as it may produce inaccurate results
#Due to the irregular shapes of the tumors
if aprox_area is not None:
#This Value is in Pixels and should be converted into some standard metric
    print(aprox_area.item())



    

        

