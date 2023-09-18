import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def imshow2(inp, title=None):
    # Display an image
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.subplot(num_images//4+1, 4, images_so_far)
                plt.axis('off')
                plt.title(class_names[preds[j]])
                imshow2(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    plt.tight_layout()    
    plt.show()

if __name__ == '__main__':
    # Data directory
    data_dir = './dataset'

    # Data transformations for testing
    data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    # Load test datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['test']}
    print(dataloaders)

    # Get class names
    class_names = image_datasets['test'].classes
    print(class_names)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the pretrained model
    model_ft = torch.load('train_by_myself.pth')
    model_ft = model_ft.to(device)

    # Set the model to evaluation mode
    model_ft.eval()

    # Use the model for prediction or evaluation
    visualize_model(model_ft)  # Make sure to pass 'test_loader' as the dataloader parameter
