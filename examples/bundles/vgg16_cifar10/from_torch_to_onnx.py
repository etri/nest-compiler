import torch
from torchvision import datasets, transforms
from vgg import VGG
import torch.onnx
#import onnxruntime

def get_data(data_dir, batch_size, test_batch_size):
    #kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs = {}
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
    criterion = torch.nn.CrossEntropyLoss()
    return train_loader, val_loader, criterion

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
data_dir = '/home/taeho/github/nest-compiler/glow/tests/images'
batch_size = 16
test_batch_size = 1

_, val_loader, criterion = get_data(data_dir, batch_size, test_batch_size)
pretrained_model_dir = './model_trained.pth' #'./tmp_model.pth'
model = VGG().to(device)
model.load_state_dict(torch.load(pretrained_model_dir))
model.eval()

x, _ = next(iter(val_loader))

#torch_out = model(x.to(device))
#torch.onnx.export(model, x.to(device), "vgg16.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
#x = torch.randn(batch_size, 1, 32, 32, required_grad=True)

torch_out = model(x)
torch.onnx.export(model, x, "vgg16_1.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
