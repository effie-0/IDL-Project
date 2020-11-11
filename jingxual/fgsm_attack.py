def testattack(net, testloader, epsilon):
    # global best_acc
    net.eval()

    # test_loss = 0
    correct = 0
    # total = 0
    # with torch.no_grad():
    for b, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs.requires_grad = True
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        # if predicted.item() != targets.item():
        #     continue
        loss = criterion(outputs, targets)
        net.zero_grad()

        # test_loss += loss.item()
        loss.backward()
        data_grad = inputs.grad.data
        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)
        new_outputs = net(perturbed_data)
        _, new_predicted = new_outputs.max(1)

        # total += targets.size(0)
        correct += new_predicted.eq(targets).sum().item()

        torch.cuda.empty_cache()
        del inputs
        del targets
        
    acc = correct/float(len(testloader))
    # avg_loss = test_loss/total
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(testloader), acc))

    return acc
