if __name__ == '__main__':    
    '''Train CINIC10 with PyTorch.'''
    import time
    startTime = time.time()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torch.backends.cudnn as cudnn

    import torchvision
    import torchvision.transforms as transforms
    from torchvision import datasets
    import os
    import argparse

    from models import *
    from utils import progress_bar


    parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data

    print('==> Preparing data..')

    cinic_directory = "../data"

    cinic_mean_RGB = (0.47889522, 0.47227842, 0.43047404)
    cinic_std_RGB = (0.24205776, 0.23828046, 0.25874835)

    # define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean_RGB, cinic_std_RGB),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean_RGB, cinic_std_RGB),
    ])

    # define datasets
    trainset = datasets.ImageFolder(cinic_directory + '/train',transform=transform_train)
    testset = datasets.ImageFolder(cinic_directory + '/test',transform=transform_test)

    # define DataLoaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19') 
    # net = ResNet18() 
    # net = GoogLeNet()
    net = DenseNet121() 
    # net = ResNeXt29_2x64d()
    # net =ResNet50()
    # net = MobileNet()

    net = net.to(device)
    if device == 'cuda':
        torch.cuda.empty_cache()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch+100):
        train(epoch)
        test(epoch)
        scheduler.step()
    exTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(exTime))
    with open('./NET_NAME_runtime_100epochs.txt','w') as f:
        f.write(f'NET_NAME total runtime with 100 epochs on full dataset is: {str(exTime)}')
    