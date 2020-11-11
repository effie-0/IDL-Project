def train(net, epoch):
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
        
        torch.cuda.empty_cache()
        del inputs
        del targets

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = correct/total
    avg_loss = train_loss/total
    
    return avg_loss, acc


def test(net, epoch):
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
            
            torch.cuda.empty_cache()
            del inputs
            del targets
            
    acc = correct/total
    avg_loss = test_loss/total
    
    return avg_loss, acc
    
def fgsm_train(net, epoch, eps=0.01):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs.requires_grad = True

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        data_grad = inputs.grad.data
        perturbed_data = fgsm_attack(inputs, eps, data_grad)
        new_outputs = net(perturbed_data)
        new_loss = criterion(new_outputs, targets)
        new_loss.backward()
        
        optimizer.step()

        train_loss += new_loss.item()
        _, new_predicted = new_outputs.max(1)
        total += targets.size(0)
        correct += new_predicted.eq(targets).sum().item()
        
        torch.cuda.empty_cache()
        del inputs
        del targets

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = correct/total
    avg_loss = train_loss/total
    
    return avg_loss, acc
