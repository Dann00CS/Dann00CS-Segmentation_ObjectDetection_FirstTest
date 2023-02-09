import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import numpy as np

def conv3x3_bn(ci, co):
    return torch.nn.Sequential(
        torch.nn.Conv2d(ci, co, 3, padding=1),
        torch.nn.BatchNorm2d(co),
        torch.nn.ReLU(inplace=True)
    )

def encoder_conv(ci, co):
  return torch.nn.Sequential(
        torch.nn.MaxPool2d(2),
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
    )

class deconv(torch.nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
        self.conv1 = conv3x3_bn(ci, co)
        self.conv2 = conv3x3_bn(co, co)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX, 0, diffY, 0))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet(pl.LightningModule):
    def __init__(self, n_classes=2, in_ch=3):
        super().__init__()

        c = [16, 32, 64, 128]

        self.conv1 = torch.nn.Sequential(
          conv3x3_bn(in_ch, c[0]),
          conv3x3_bn(c[0], c[0]),
        )

        self.conv2 = encoder_conv(c[0], c[1])
        self.conv3 = encoder_conv(c[1], c[2])
        self.conv4 = encoder_conv(c[2], c[3])

        self.deconv1 = deconv(c[3],c[2])
        self.deconv2 = deconv(c[2],c[1])
        self.deconv3 = deconv(c[1],c[0])

        self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)

        x = self.deconv1(x, x3)
        x = self.deconv2(x, x2)
        x = self.deconv3(x, x1)
        x = self.out(x)
        return x

    def iou(self, outputs, labels):
        # aplicar sigmoid y convertir a binario
        outputs, labels = torch.sigmoid(outputs) > 0.5, labels > 0.5
        SMOOTH = 1e-6
        # BATCH x num_classes x H x W
        B, N, H, W = outputs.shape
        ious = []
        for i in range(N-1): # saltamos el background
            _out, _labs = outputs[:,i,:,:], labels[:,i,:,:]
            intersection = (_out & _labs).float().sum((1, 2))  
            union = (_out | _labs).float().sum((1, 2))         
            iou = (intersection + SMOOTH) / (union + SMOOTH)  
            ious.append(iou.mean().item())
        return np.mean(ious) 

    def predict(self, x):
        with torch.no_grad():
          output = self(x)
          # print(y_hat)
          return torch.argmax(output, axis=0)

    def compute_loss_and_metrics(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        iou_metric = self.iou(y_hat, y)
        return loss, iou_metric

    def training_step(self, batch, batch_idx):
        loss, iou_metric = self.compute_loss_and_metrics(batch)
        self.log('loss', loss)
        self.log('iou', iou_metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou_metric = self.compute_loss_and_metrics(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou_metric, prog_bar=True)
        # return loss, iou_metric

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer