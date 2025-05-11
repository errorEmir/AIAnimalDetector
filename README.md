
## ⚙️ Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ererorEmir/AIAnimalDetector.git
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```
    *   **Windows:** `.\.venv\Scripts\activate`
    *   **Linux/macOS:** `source .venv/bin/activate`

3.  **Install Dependencies:**
    *   **Important:** First, install the correct PyTorch version with CUDA support for your system. Get the right command from the [PyTorch Installation Page](https://pytorch.org/get-started/locally/). Example (for CUDA 11.8):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   Then, install the other dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        *(If you don't have a `requirements.txt` file, create one using `pip freeze > requirements.txt` and keep only the necessary packages: torch, torchvision, gradio, Pillow, numpy, scikit-learn, kaggle).*

4.  **Download the Dataset:**
    *   **Method 1: Using Kaggle API (Recommended)**
        *   Set up your Kaggle API key (`kaggle.json`) (See [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api#api-credentials)).
        *   Run the following commands in the project's root directory:
            ```bash
            pip install kaggle # If not already installed
            kaggle datasets download -d alessiocorrado99/animals10 -p . --unzip
            # Ensure the downloaded files end up directly in a 'raw-img' folder
            # or move the extracted 'raw-img' folder to the project root.
            # You can delete the zip file: rm animals10.zip (Linux/macOS) or del animals10.zip (Windows)
            ```
    *   **Method 2: Manual Download**
        *   Download the dataset from the [Animals-10 Kaggle Page](https://www.kaggle.com/datasets/alessiocorrado99/animals10).
        *   Extract the downloaded ZIP file.
        *   Copy the `raw-img` folder into the project's root directory.

    *The final structure should have the `raw-img` folder in the project root, containing the 10 animal subfolders.*

## 🚀 Usage

1.  **Train the Model (Optional - if a pre-trained model exists):**
    *   To train the model from scratch or retrain it, run the following command:
        ```bash
        python train.py
        ```
    *   This process will train the model using the data in the `raw-img` folder and save the results (`animal_classifier_resnet18.pth` and `class_names.json`) into the `models/` directory. Training time varies depending on your hardware (especially GPU availability).

2.  **Launch the Web Application:**
    *   After the model is trained (or if a pre-trained model exists in the `models/` folder), start the Gradio interface with the command:
        ```bash
        python app.py
        ```
    *   Open the local address provided in the terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`) in your web browser.
    *   Upload an animal image through the interface and click the "Submit" (or similar) button to see the prediction.

## 📊 Model Performance

The model achieved the following performance on the validation set (20% of the Animals-10 dataset):

*   **Validation Accuracy:** `XX.XX%` <!-- Enter the best accuracy value from the train.py output here -->
*   **Validation Precision (Macro):** `0.XX` <!-- Enter the value from the train.py output here -->
*   **Validation Recall (Macro):** `0.XX` <!-- Enter the value from the train.py output here -->
*   **Validation F1-Score (Macro):** `0.XX` <!-- Enter the value from the train.py output here -->

*(These metrics might correspond to the epoch with the best validation accuracy or the last epoch. Check the `train.py` output for details.)*

## 🖼️ Screenshots
(./screenshot.png) 
(./result.png) 

---

*This README file was generated based on the project requirements and code structure.*
