import os
import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.onnx as onnx

from model import Net
from azureml.core import Run

run = Run.get_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the training data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for SGD'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD'
    )

    args = parser.parse_args()

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    # prepare DataLoader for CIFAR10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=False,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    # define convolutional network
    net = Net()

    # set up pytorch loss /  optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )

    # train the network
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # unpack the data
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                loss = running_loss / 2000
                run.log('loss', loss)  # log loss metric to AML
                print(f'epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}')
                running_loss = 0.0

    print('Finished Training')





    os.makedirs('outputs', exist_ok=True)
 
    torch.save(net,'outputs/cifar10_net.pt')

    # torch.save(net.state_dict(),'outputs/cifar10_net.pt')
    # model=Net()
    # model.load_state_dict(torch.load('outputs/cifar10_net.pt'))
    # model.eval()

    # dummy_input=torch.randn(1,3,32,32)
    # export_file="outputs/cifar10_net.onnx"
    # torch.onnx.export(model,dummy_input,export_file,input_names=["input"],output_names=["output"])

    







    