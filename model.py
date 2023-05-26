'''
adapted from https://github.com/fungtion/DANN_py3/blob/master/model.py
by James Kim
May 25, 2023
'''
import torch.nn as nn
from function import GradientReversalLayerF as GRL

class DDAModel(nn.Module): #Deep Domain Adaptation
    def __init__(self):
        super(DDAModel, self).__init__()
        # Feature Extractor
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5)) # 3 channel / MNIST: (28,28) -> (24, 24)
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2)) # MNIST: (24,24) -> (12, 12)
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5)) # MNIST: (12,12) -> (8, 8)
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2)) # MNIST: (8, 8) -> (4, 4)
        self.feature.add_module('f_relu2', nn.ReLU(True))

        # Label Predictor
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('y_fc1', nn.Linear(50 * 4 * 4, 100))
        self.label_predictor.add_module('y_bn1', nn.BatchNorm1d(100))
        self.label_predictor.add_module('y_relu1', nn.ReLU(True))
        self.label_predictor.add_module('y_drop1', nn.Dropout())
        self.label_predictor.add_module('y_fc2', nn.Linear(100, 100))
        self.label_predictor.add_module('y_bn2', nn.BatchNorm1d(100))
        self.label_predictor.add_module('y_relu2', nn.ReLU(True))
        self.label_predictor.add_module('y_fc3', nn.Linear(100, 10))
        self.label_predictor.add_module('y_softmax', nn.LogSoftmax(dim = 1))

        # Domain Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear( 50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100,2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, lamda):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4  *4)
        reverse_feature = GRL.apply(feature, lamda)
        label_output = self.label_predictor(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output