import abc
import cv2
import torchvision.transforms as T
import albumentations as alb
from albumentations.pytorch import ToTensor
import numpy as np

class AlbumentationTrans:
    
    def __init__(self, transform):
        self.album_transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.album_transform(image=img)['image']

class AugmentationBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class FLOWERS102_Transforms(AugmentationBase):

    def build_train(self):
        return T.Compose([
              T.ToTensor(),
              T.RandomRotation((-15., 15.), fill=0),
              T.Resize((256, 256)),
              T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
              T.RandomApply([T.CenterCrop(224), ], p=0.5),
              # transforms.ElasticTransform(),
              T.Resize((224, 224)),
              # T.RandomErasing(p=1, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random')

        ])

    def build_test(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

class Flowers102_AlbumTrans(AugmentationBase):

    def __init__(self, augs= None):
      self.augs = augs

    def build_train(self):
        
        if self.augs is None:
          train_trans = alb.Compose([
                                     
              # alb.PadIfNeeded(40,40,cv2.BORDER_CONSTANT,[4,4],0),
              alb.Rotate(limit=15, p=1.0),
              alb.Resize(height=256, width=256),
              alb.Normalize(
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]
              ),
              alb.CenterCrop(224,224, always_apply= True),
              alb.HorizontalFlip(p= 0.75),
              # alb.CoarseDropout(
              #     max_holes=1, max_height=50, max_width=50, min_holes=1, min_height=50, min_width=50, 
              #     fill_value=[0.485*255, 0.456*255, 0.406*255], always_apply=False, p=0.75
              # ),

              ToTensor()
          ])
        else:
          train_trans = alb.Compose(self.augs)
        
        return AlbumentationTrans(train_trans)

    def build_test(self):
        test_trans = alb.Compose([
            alb.Resize(height=224, width=224),
            alb.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensor()
        ])
        return AlbumentationTrans(test_trans)
