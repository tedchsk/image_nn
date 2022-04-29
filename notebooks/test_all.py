# from ResNet import ResNet
# from DenseNet import DenseNet
# from DSNet import DSNet
from core.model import ResNet, DenseNet, DSNet


def test_ResNet():
    model = ResNet(model_n=3, num_classes=100, device="cpu")


def test_DenseNet():
    # Just checking that the model can be created with no errors
    model_n = 3
    model = DenseNet(model_n=2)


def test_DSNet():
    DSNet(model_n=3, num_classes=100, device="cpu")
