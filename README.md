# petdisease-detector 
PetVision — Pet Skin Condition Detector
PetVision is a deep learning-based pet skin condition classifier that detects 6 common skin conditions in dogs from images using a fine-tuned ResNet50 model.

🔍 Conditions Detected
Condition	Description
Dermatitis	Skin inflammation
Fungal Infection	Fungal skin disease
Healthy	No condition detected
Hypersensitivity	Allergic skin reactions
Demodicosis	Mite-caused skin condition
Ringworm	Fungal ring-shaped infection

🧠 Model
Architecture: ResNet50 (Transfer Learning)
Framework: PyTorch
Input: Pet skin image (JPG/PNG)
Output: Predicted condition + confidence score


📁 Project Structure

petvision/
├── api.py          # FastAPI server
├── model.py        # ResNet50 model definition
├── predict.py      # Prediction engine
├── train.py        # Training script
├── dataset.py      # Dataset loader
├── requirments.txt # Dependencies
└── best_model.pth  # Model weights (download separately)
