from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import base64
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

application = Flask(__name__)
CORS(application)

class MNIST_Neural_Network(nn.Module):
    def __init__(self, in_layer=784, hid_layer1=800, hid_layer2=130, hid_layer3=80, out_layer=10):
        super().__init__()
        self.w1 = nn.Linear(in_layer, hid_layer1)
        self.w2 = nn.Linear(hid_layer1, hid_layer2)
        self.w3 = nn.Linear(hid_layer2, hid_layer3)
        self.out = nn.Linear(hid_layer3, out_layer)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.w1(x))
        x = F.relu(self.w2(x))
        x = F.relu(self.w3(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

model_path = './models/mnist_model.pth'
mnist_model = MNIST_Neural_Network()
mnist_model.load_state_dict(torch.load(model_path))
mnist_model.eval()

@application.route('/', methods=['GET', 'POST'])
def receive_image():
    if request.method == 'POST':
        data = request.get_json()
        data = data.get('image')

        image_data = base64.b64decode(data.split(',')[1])
        img = Image.open(io.BytesIO(image_data))

        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.PILToTensor(),
        ])

        img_tensor = transform(img).view(1, 28*28).float()
        img_tensor = F.normalize(img_tensor)

        with torch.no_grad():
            output = mnist_model(img_tensor)
            prediction = torch.argmax(output).item()

        return jsonify(str(prediction))

    return jsonify({'message': 'Default message'})

if __name__ == "__main__":
    application.run()
