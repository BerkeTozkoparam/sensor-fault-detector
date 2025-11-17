# Predictive Maintenance Model for Sensor Fault Detection

## Project Overview

This repository contains a machine learning project designed to predict equipment failure types based on sensor data. The model is trained on the `sensor_maintenance_data.csv` dataset to perform multi-class classification, identifying whether a sensor's readings indicate normal operation (`None`), an `Overload` condition, or an `Overheating` condition.

The project includes the complete workflow from data preprocessing to model training, serialization (saving the model), and deployment via an interactive **Gradio web interface**.

---

## Features

* **Multi-Class Classification:** Moves beyond simple binary fault detection to classify the *specific type* of failure.
* **End-to-End Pipeline:** Utilizes a `scikit-learn` `Pipeline` to bundle preprocessing (scaling, encoding) and modeling (`RandomForestClassifier`) into a single, deployable object.
* **Interactive Interface:** A user-friendly web UI built with `Gradio` allows for real-time predictions by manually inputting 17 different sensor and equipment status values.
* **Visual Feedback:** The interface provides clear, actionable results, including a "Risk Status" and a bar plot visualizing the model's confidence for each potential failure type.

---

## Application Interface Preview

Here is a preview of the Gradio application dashboard, showing the input fields and the predictive output (risk status and probability graph).

![Gradio Interface Preview](<img width="1455" height="822" alt="Ekran Resmi 2025-11-17 21 57 53" src="https://github.com/user-attachments/assets/d0c56cfa-4159-42b8-a3bf-47e27c842673" />
)

---

## Methodology

The project follows a standard data science workflow:

1.  **Data Ingestion & Cleaning:** The raw `sensor_maintenance_data.csv` is loaded. Data leakage is prevented by dropping columns that would not be available at the time of prediction (e.g., `Repair Time (hrs)`, `Fault Detected`).
2.  **Preprocessing:** A `ColumnTransformer` is used to apply different transformations to numerical and categorical data simultaneously:
    * **Numerical Features:** Standardized using `StandardScaler`.
    * **Categorical Features:** Converted into numerical format using `OneHotEncoder`.
3.  **Model Training:** A `RandomForestClassifier` is trained on the preprocessed data. The model is optimized for multi-class classification and balanced to handle class disparities.
4.  **Model Serialization:** The entire `Pipeline` (preprocessor + model) and the `LabelEncoder` (for target classes) are saved to disk as `.joblib` files for later use.
5.  **Deployment:** A `Gradio` application loads the saved `.joblib` files to serve predictions via an interactive dashboard.

---

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd sensor-fault-detector
    ```

2.  **Install Dependencies:**
    ```bash
    pip install gradio pandas "scikit-learn[full]"
    ```

3.  **Run the Application:**
    * Ensure the `.joblib` files (`sensor_model_pipeline.joblib`, `label_encoder.joblib`) are in the same directory.
    * Execute the Python script or Colab notebook containing the `Gradio` application code.
    ```bash
    python app.py
    ```

4.  **Test the Model:**
    * Open the public URL provided by Gradio in your browser.
    * Fill in the 17 input fields in the interface.
    * Click "Submit" to receive the "Risk Status" and the probability distribution graph.

---

## Project File Structure
