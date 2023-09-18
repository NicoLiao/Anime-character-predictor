import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms

class_names = ['aika', 'aisaka', 'akihime', 'akiyama', 'akizuki', 'alice', 'allen', 'amami', 'ana', 'andou', 'arcueid', 'asagiri', 'asahina', 'asakura', 'ayanami', 'ayasaki', 'belldandy', 'black', 'c.c', 'canal', 'caro', 'chii', 'cirno', 'corticarte', 'daidouji', 'enma', 'erio', 'fate', 'feena', 'flandre', 'fujibayashi', 'fukuzawa', 'furude', 'furukawa', 'fuyou', 'golden', 'hakurei', 'hatsune', 'hayama', 'hayase', 'hiiragi', 'hinamori', 'hirasawa', 'horo', 'houjou', 'ibuki', 'ichinose', 'ikari', 'illyasviel', 'ito', 'izayoi', 'izumi', 'kagamine', 'kagurazaka', 'kallen', 'kamikita', 'kanu', 'katagiri', 'katsura', 'kawashima', 'kikuchi', 'kinomoto', 'kirisame', 'kisaragi', 'kobayakawa', 'kochiya', 'koizumi', 'komaki', 'konpaku', 'kotegawa', 'kotobuki', 'kousaka', 'kururugi', 'kusugawa', 'kyon', 'lala', 'lelouch', 'li', 'lisianthus', 'louise', 'maka', 'maria', 'matou', 'matsuoka', 'megurine', 'melon-chan', 'midori', 'milfeulle', 'minamoto', 'minase', 'miura', 'miyafuji', 'miyamura', 'mizunashi', 'nagase', 'nagato', 'nagi', 'nakano', 'nanael', 'natsume', 'nerine', 'nia', 'nijihara', 'nogizaka', 'noumi', 'nunnally', 'ogasawara', 'okazaki', 'pastel', 'patchouli', 'primula', 'ranka', 'reina', 'reinforce', 'reisen', 'remilia', 'rollo', 'ryougi', 'ryuuguu', 'saber', 'saigyouji', 'sairenji', 'sakagami', 'sakai', 'sanzenin', 'saotome', 'sendou', 'seto', 'setsuna', 'shameimaru', 'shana', 'sheryl', 'shidou', 'shigure', 'shihou', 'shindou', 'shinku', 'shirakawa', 'shirley', 'shirou', 'siesta', 'sonozaki', 'sonsaku', 'souryuu', 'subaru', 'suigintou', 'suzumiya', 'tainaka', 'takamachi', 'takara', 'takatsuki', 'teana', 'tohsaka', 'tsukimura', 'tsuruya', 'vita', 'vivio', 'yagami', 'yakumo', 'yoko', 'yoshida', 'yuno', 'yuuki', 'yuzuhara']

def upload_and_predict():
    # Use filedialog to let the user select an image file
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and display the image
    image = Image.open(file_path).convert('RGB')
    image = image.resize((150, 150))  # Resize the image to fit the GUI
    tk_image = ImageTk.PhotoImage(image)
    label_image.config(image=tk_image)
    label_image.image = tk_image

    # Use the model to predict the image class
    predicted_class = predict_single_image(model_ft, image, device)
    label_result.config(text=f"Predicted class: {predicted_class}")

def predict_single_image(model, image, device):

    # Apply the same preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)

    if image_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 channels in the image tensor, but got {image_tensor.shape[0]} channels")

    # Expand to a batch
    image_tensor = image_tensor.unsqueeze(0)

    # Predict
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted[0]]

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = torch.load('anime.pth')
    model_ft = model_ft.to(device)
    model_ft.eval()

    # GUI (Graphic User Interface)
    window = tk.Tk()
    window.geometry("800x600")
    window.title("Anime Character Predictor")

    # Upload button
    btn_upload = tk.Button(window, text="Upload an image", command=upload_and_predict)
    btn_upload.pack(pady=20)

    # Image display label
    label_image = tk.Label(window)
    label_image.pack(pady=20)

    # Predicted result display label
    label_result = tk.Label(window, text="Predicted class will appear here", font=("Arial", 12))
    label_result.pack(pady=20)

    window.mainloop()
