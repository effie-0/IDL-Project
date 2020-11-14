import torch 


def fgsm_attack(image, epsilon, data_grad, mask = None):
    '''
    image: batch x 3 x 32 x 32
    
    data_grad: batch x 3 x 32 x32
    
    mask: batch_size x 1 x 1 x 1, 1 for false prediction, 0 for correct prediction, use for accelarate computation
    '''
    if mask is None:
        sign_data_grad = data_grad.sign()
    else:
        # Collect the element-wise sign of the data gradient
        sign_data_grad = torch.mul(data_grad.sign(), mask.view(-1, 1, 1, 1))

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image


