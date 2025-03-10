# Bone-Fracture-X-ray-Classification
## Overview
This project implements a deep learning model using Convolutional Neural Networks (CNNs) for classifying X-ray images of bone fractures into Simple and Comminuted fracture types. The model is trained and evaluated using a publicly available dataset.

## Dataset
The dataset used for this project is sourced from Mendeley Data:
[Bone Fracture X-ray Dataset (Simple vs. Comminuted)](https://data.mendeley.com/datasets/vg95gvhj3y/3)

## Project Structure
```
X_ray_Bone_Fracture_Classification/
│── data/                 # Directory to store dataset (not included in repo)
│── models/               # Directory for trained models
│── X_ray_Bone_Fracture_Classification.ipynb  # Jupyter Notebook with implementation
│── requirements.txt      # Python dependencies
│── LICENSE              # Open-source MIT License
│── README.md            # Project documentation
```

## Installation & Setup
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/X_ray_Bone_Fracture_Classification.git
   cd X_ray_Bone_Fracture_Classification
   ```

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Visit the dataset link provided above.
   - Download and extract the dataset inside the `data/` directory.

4. **Run the Jupyter Notebook**:
   - Open Jupyter Notebook:
     ```sh
     jupyter notebook
     ```
   - Open `X_ray_Bone_Fracture_Classification.ipynb` and run the cells.

## Model Architecture
The implemented CNN consists of:
- Three convolutional layers (with ReLU activations and max pooling)
- Two fully connected layers
- Output layer for binary classification (Simple vs. Comminuted fractures)

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Training & Evaluation
### Training Results
The model was trained for **10 epochs** with the following results:
```
Epoch [1/10], Loss: 0.7273, Accuracy: 57.05%
Epoch [2/10], Loss: 0.6262, Accuracy: 64.30%
Epoch [3/10], Loss: 0.5480, Accuracy: 70.43%
Epoch [4/10], Loss: 0.4719, Accuracy: 76.34%
Epoch [5/10], Loss: 0.3813, Accuracy: 81.42%
Epoch [6/10], Loss: 0.2961, Accuracy: 87.16%
Epoch [7/10], Loss: 0.1976, Accuracy: 92.41%
Epoch [8/10], Loss: 0.1162, Accuracy: 95.76%
Epoch [9/10], Loss: 0.0688, Accuracy: 98.07%
Epoch [10/10], Loss: 0.0238, Accuracy: 99.33%
```

### Testing Results
The model achieved the following performance on the test dataset:
```
Test Loss: 0.0089, Test Accuracy: 99.87%
```

## Applications
- **Medical Diagnosis Assistance**: Helps radiologists identify and classify bone fractures more accurately.
- **AI-powered Healthcare**: Can be integrated into AI-based diagnostic tools to provide automated fracture detection.
- **Educational Purposes**: Useful for training medical students in identifying different types of fractures.

## Future Scope
- **Increase dataset diversity**: Training on more diverse X-ray datasets can improve robustness.
- **Fine-tuning with pre-trained models**: Using pre-trained architectures like ResNet or DenseNet can enhance accuracy.
- **Deployment as a web application**: Can be converted into a cloud-based AI service for real-time fracture classification.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Developed by Umar Zafar for a client project.

---
**Note:** This implementation was developed and executed using **Google Colab**.
