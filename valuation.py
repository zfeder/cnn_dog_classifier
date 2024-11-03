from library import *
from model import *
from preprocessing import *

transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),        
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

img1 = Image.open('cat.jpg')
img2 = Image.open('dog/dog.jpg')
img3 = Image.open('dog/dog1.jpg')
img4 = Image.open('dog/dog2.jpg')
img5 = Image.open('dog/dog3.jpg')


img1_tensor = transform(img1)
img2_tensor = transform(img2)
img3_tensor = transform(img3)
img4_tensor = transform(img4)
img5_tensor = transform(img5)


batch_of_images = torch.stack([img1_tensor, img2_tensor, img3_tensor, img4_tensor, img5_tensor]) 


cnn.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_of_images = batch_of_images.to(device)
cnn = cnn.to(device)


with torch.no_grad():  
    output = cnn(batch_of_images)

predictions = (output > 0.5).float() 


for i, prob in enumerate(output):
    percent = prob.item() * 100
    label = "Dog" if predictions[i].item() == 1 else "Not dog"
    if label == "Dog":
        print(f"Image {i+1}: {percent:.2f}% probability that is a {label}")
    else:
        print(f"Image {i+1}: {(100 - percent):.2f}% probability that is a {label}")


def denormalize(tensor):
    return tensor * 0.5 + 0.5 


fig, axes = plt.subplots(1, 5, figsize=(15, 5))  
for i, ax in enumerate(axes):
    ax.imshow(transforms.ToPILImage()(batch_of_images[i].cpu())) 
    ax.set_title(f"Prediction: {'Dog' if predictions[i].item() == 1 else 'Not dog'}")
    ax.axis('off')  
plt.show()


fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    img_denorm = denormalize(batch_of_images[i].cpu())  
    ax.imshow(transforms.ToPILImage()(img_denorm))
    ax.set_title(f"Prediction: {'Dog' if predictions[i].item() == 1 else 'Not dog'}")
    ax.axis('off')
plt.show()
