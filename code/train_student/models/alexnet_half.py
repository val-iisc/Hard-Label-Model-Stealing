import torch 
import torch.nn as nn

class AlexNet_half(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_half, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = nn.ReLU()        
        self.lrn = nn.LocalResponseNorm(2)        
        self.pad = nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = nn.BatchNorm2d(24, eps=0.001)
        
        self.conv2 = nn.Conv2d(24, 64, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = nn.BatchNorm2d(64, eps=0.001)
        
        self.conv3 = nn.Conv2d(64, 96, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = nn.BatchNorm2d(96, eps=0.001)
        
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = nn.BatchNorm2d(96, eps=0.001)
        
        self.conv5 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = nn.BatchNorm2d(64, eps=0.001)
        
        self.fc1 = nn.Linear(576,256)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = nn.Dropout(p=0.5)
        
        self.batch_norm6 = nn.BatchNorm1d(256, eps=0.001)
        
        self.fc2 = nn.Linear(256,128)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = nn.BatchNorm1d(128, eps=0.001)
        
        self.fc3 = nn.Linear(128,10)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        self.soft = nn.Softmax()
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 64*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        #softmax_val = self.soft(logits)

        return logits

model = AlexNet_half(10)
