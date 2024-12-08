import torch
import torchvision.transforms as transforms
from PIL import Image
import model.model as module_arch  # Ensure model architecture can be accessed

# Load model function
def load_model(checkpoint_path, config):
    model = config.init_obj('arch', module_arch)  # Initialize the model
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
    model.eval()  # Set to evaluation mode
    return model

# Preprocessing function for the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model's expected input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
    ])
    image = Image.open(image_path).convert('RGB')  # Open image
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction function
def predict(image_path, model, device):
    image = preprocess_image(image_path).to(device)  # Load and preprocess image
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)  # Get the predicted class (0 or 1 for benign/malignant)
    return 'Malignant' if pred.item() == 1 else 'Benign'

if __name__ == "__main__":
    # Configuration and paths
    checkpoint_path = 'saved/models/BCDensenet/1026_223811/checkpoint-epoch15.pth'  # Path to model checkpoint
    config = ConfigParser.from_args()  # Load config if needed, or create manually

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, config).to(device)

    # Provide image path for prediction
    image_path = 'path/to/your/histopathological_image.jpg'
    result = predict(image_path, model, device)
    print(f'Prediction for the image is: {result}')
