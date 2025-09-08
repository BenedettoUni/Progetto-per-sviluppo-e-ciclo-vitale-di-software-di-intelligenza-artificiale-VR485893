import torchvision
import torchvision.transforms as transforms

def get_dataset():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return train_set

def test_dataset_length():
    ds = get_dataset()
    assert len(ds) > 0, "Dataset vuoto!"

def test_dataset_classes():
    ds = get_dataset()
    assert len(ds.classes) == 10, "Il dataset CIFAR-10 deve avere 10 classi"

def test_image_shape():
    ds = get_dataset()
    x, y = ds[0]
    assert x.shape == (3, 32, 32), f"Shape errata: {x.shape}"
    assert 0 <= y < 10, f"Label fuori range: {y}"