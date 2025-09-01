import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Nomi delle classi di CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Trasformazioni di base (tensor + normalizzazione)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carico dataset CIFAR-10
train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Uso solo 30.000 immagini (25k train, 5k test)
train_set.data = train_set.data[:25000]
train_set.targets = train_set.targets[:25000]

test_set.data = test_set.data[:5000]
test_set.targets = test_set.targets[:5000]

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# Report veloce
print("CIFAR-10 EDA")
print("Train size:", len(train_set))
print("Test size:", len(test_set))

# Distribuzione delle classi (grafico a barre)
labels = np.array(train_set.targets)
unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(10,4))
sns.barplot(x=[class_names[i] for i in unique], y=counts)
plt.title("Distribuzione delle classi (train)")
plt.xticks(rotation=45)
plt.show()

# Alcune immagini di esempio
examples = enumerate(train_loader)
_, (imgs, targets) = next(examples)

plt.figure(figsize=(10,6))
for i in range(10):
    img = imgs[i] / 2 + 0.5   # "unnormalize"
    npimg = img.numpy().transpose((1, 2, 0))
    plt.subplot(2,5,i+1)
    plt.imshow(npimg)
    plt.title(class_names[targets[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
