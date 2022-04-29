# from ResNet import ResNet
# from DenseNet import DenseNet
# from DSNet import DSNet
from model import ResNet, DenseNet, DSNet

def test_ResNet():
    model = ResNet(model_n=3, num_classes=100, device="cpu")

def test_DenseNet():
    # Just checking that the model can be created with no errors
    model_n = 3
    model = DenseNet(growth_rate=16, block_config=(2 * model_n, 2 * model_n, 2 * model_n),
                 num_init_features=16, bn_size=2, num_classes=100)
  
def test_DSNet():
    DSNet(model_n=3, num_classes=100, device="cpu")
