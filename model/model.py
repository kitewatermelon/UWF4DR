import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy, F1Score, AUROC, Precision, Recall
from config import NUM_CLASSES

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(pl.LightningModule):
    def __init__(self, block, layers, num_classes=NUM_CLASSES, learning_rate=1e-3):
        super(ResNet, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters like num_classes and learning_rate
        self.in_channels = 64
        self.learning_rate = learning_rate

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.valid_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.f1_score = F1Score(num_classes=1, task='binary', average='none')
        self.aucroc = AUROC(num_classes=1, task='binary')
        self.precision = Precision(num_classes=1, task='binary', average='none')
        self.recall = Recall(num_classes=1, task='binary', average='none')
        
        # Define network layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(x, y)
        
        # Compute metrics
        accuracy = self.train_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        aucroc = self.aucroc(scores, y)
        precision = self.precision(scores, y)
        recall = self.recall(scores, y)

        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_aucroc': aucroc,
            'train_precision': precision,
            'train_recall': recall,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss, 'scores': scores, 'y': y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(x, y)

        # Compute metrics
        accuracy = self.valid_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        aucroc = self.aucroc(scores, y)
        precision = self.precision(scores, y)
        recall = self.recall(scores, y)

        self.log_dict({
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_f1_score': f1_score,
            'val_aucroc': aucroc,
            'val_precision': precision,
            'val_recall': recall,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss, 'scores': scores, 'y': y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(x, y)
        
        # Compute metrics
        accuracy = self.test_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        aucroc = self.aucroc(scores, y)
        precision = self.precision(scores, y)
        recall = self.recall(scores, y)

        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_aucroc': aucroc,
            'test_precision': precision,
            'test_recall': recall,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss, 'scores': scores, 'y': y}

    def _common_step(self, x, y):
        # Forward pass through the model
        scores = self.forward(x)
        
        # Apply sigmoid to get probabilities for binary classification
        scores = torch.sigmoid(scores)
        
        # Convert probabilities to binary predictions
        binary_scores = (scores >= 0.5).long()
        binary_scores = binary_scores.squeeze(1)  # Remove any extra dimensions

        # Calculate the loss
        loss = self.loss_fn(scores.squeeze(1), y)  
        y = y.to(dtype=torch.int)  # Ensure y is of integer type for metrics
        return loss, binary_scores, y

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Instantiate ResNet-34 using the LightningModule structure
def resnet34(num_classes=NUM_CLASSES, learning_rate=1e-3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, learning_rate)
