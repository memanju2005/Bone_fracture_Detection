# B.Tech Final Year Major Project Report

## Project Title: Bone Fracture Detection using Deep Learning

### Abstract
Bone fractures have historically been a significant medical issue, and their diagnosis via X-ray images has heavily relied on human expertise, which can sometimes be subjective or prone to error. In recent years, Machine Learning and Deep Learning-based computer vision solutions have revolutionized the medical field by offering automated, high-precision diagnostics. This project aims to develop an intelligent and feasible deep-learning solution for the identification and classification of various bone fractures. Utilizing Convolutional Neural Networks (CNNs), specifically the transfer-learning architecture ResNet50, we have developed a hierarchical two-step classification model. The system first identifies the anatomical part of the bone (Elbow, Hand, or Shoulder) and then detects whether a fracture is present. The system achieves commendable accuracy on the MURA dataset and is wrapped in a user-friendly graphical interface, demonstrating its potential as an assistive diagnostic tool for medical practitioners.

---

### 1. Introduction
With the rapid integration of Artificial Intelligence (AI) in healthcare, medical imaging analysis has seen substantial improvements. The traditional method of fracture detection requires a radiologist to manually inspect X-ray scans, a process that can be time-consuming and sometimes challenging for subtle fractures. 

Our project focuses on creating an AI-based system that acts as a second pair of eyes for doctors. By fine-tuning the pre-trained modern CNN architecture—ResNet50—we have developed a robust pipeline capable of classifying bone types and subsequently identifying positive (fractured) or negative (normal) conditions.

---

### 2. Dataset Description
The dataset used for this project is **MURA** (musculoskeletal radiographs), one of the largest public radiographic image datasets. 
Our subset contains over 20,000 images focused on three specific upper extremity parts:
- **Elbow**
- **Hand**
- **Shoulder**

#### Data Distribution:
| **Part**     | **Normal (Negative)** | **Fractured (Positive)** | **Total** |
|--------------|:--------------------:|:-----------------------:|:---------:|
| **Elbow**    |          3160        |            2236         |    5396   |
| **Hand**     |          4330        |            1673         |    6003   |
| **Shoulder** |          4496        |            4440         |    8936   |

The data is separated into training and validation sets at the patient level, ensuring that images from the same patient are not split across training and testing environments, which prevents data leakage.

---

### 3. Proposed Methodology and System Architecture

Our methodology employs a dual-stage deep learning pipeline leveraging transfer learning. Given the relatively small size of medical datasets compared to standard computer vision datasets like ImageNet, transfer learning using **ResNet50** is highly effective. The model weights are pre-trained on ImageNet, allowing the network to leverage learned hierarchical features (like edges, curves, and textures) out of the box.

#### 3.1. Data Preprocessing & Augmentation
- The images are resized to a uniform dimension of `224x224` pixels.
- The pixel values are preprocessed using `tf.keras.applications.resnet50.preprocess_input`.
- Data augmentation, specifically horizontal flipping, is applied to the training dataset to increase dataset robustness and reduce over-fitting.
- The overall dataset is randomly split: **72% Training, 18% Validation, and 10% Testing**.

#### 3.2. Two-Step Classification Architecture
1. **Bone Type Identification (BodyPart Model):** 
   - When an X-ray is uploaded, the first ResNet50 model (`ResNet50_BodyParts.h5`) infers which part of the body is in the image (Elbow, Hand, or Shoulder).
2. **Fracture Detection (FractureSpecific Model):**
   - Based on the predicted body part, a specialized, independently trained ResNet50 model is loaded (e.g., `ResNet50_Elbow_frac.h5`).
   - This specialized model then performs a binary classification to predict if the specific bone is `Normal` or `Fractured`.

#### 3.3. Training Strategy
- Base model weights are frozen (`trainable = False`) to act as feature extractors.
- Global Average Pooling is utilized instead of Flatten layers to minimize parameters and prevent overfitting.
- Fully connected Dense layers (`128` neurons -> `50` neurons -> Output layer) are added on top of the ResNet50 features.
- The `Adam` optimizer is used with a learning rate of `0.0001` alongside `Categorical Crossentropy` loss.
- `EarlyStopping` callbacks are applied restoring the best weights to avoid vanishing gradients/over-fitting.

---

### 4. Implementation Details
The project is implemented in Python and incorporates several domains of software engineering:
1. **Machine Learning Pipeline:** Built primarily with `TensorFlow` and `Keras`. `Scikit-learn` was utilized for train-test splitting. `Pandas` was used to manipulate dataset paths and labels via DataFrames.
2. **Visualization:** `Matplotlib` is used for visualizing training/validation accuracy and loss over epochs.
3. **Graphical User Interface (GUI):** A web application was developed using `Streamlit` to provide a modern, easy-to-use, and accessible interface. Additionally, a desktop application was developed using `customtkinter` and `tkinter`. Both interfaces support an "upload and predict" flow and provide options to save screenshot results with `PyAutoGUI` and `PyGetWindow`.

**Requirements / Dependencies:**
- Python 3.7+
- TensorFlow ~2.6.2, Keras ~2.6.0
- Numpy, Pandas, Matplotlib, Scikit-learn
- Streamlit, customTkinter, Pillow, PyAutoGUI, PyGetWindow

---

### 5. Results and Discussion

Through training the specific isolated models, the classification has yielded promising results. By training distinct models tailored to distinct anatomy (instead of one giant model detecting both body part and fracture), the models isolate fracture features more clearly.

#### Evaluation Metrics:
During evaluation on the 10% separate testing set, it was observed that the models converged smoothly.
- **Accuracy & Loss:** Training procedures plateaued favorably without severe divergence between training and validation loss, thanks to the deployment of the `EarlyStopping` strategy (patience=3).
- **Body Part Detection:** The primary body part classification yielded highly accurate predictions, seamlessly funneling the respective image to its specific sub-model handler.
- **Fracture Evaluation (Elbow, Hand, Shoulder):** Demonstrated strong precision and recall handling MURA’s challenging sub-images.

Visualizations of loss and accuracy across epochs can be observed via the stored matplotlib graphs in the `plots/` directory of the repository. Further fine-tuning and data collection (such as clinical metadata) could elevate these base evaluation metrics closer to 100%. 

---

### 6. Conclusion and Future Work

#### Conclusion
This project successfully implements an end-to-end deep learning diagnostic assistant. The two-tier ResNet50-based architecture proves more efficient than a single multi-class model. The pipeline automates the extraction of complex radiological structures from X-ray inputs, resulting in a system capable of accurately flagging bone fractures in hands, shoulders, and elbows. 

#### Future Work
- **Expand Anatomical Classes:** Extend the dataset to include lower extremities such as legs, knees, ankles, and the skull.
- **Explainable AI (Grad-CAM):** Integrate heatmaps into the GUI to highlight exactly *where* on the X-ray the model suspects the fracture resides.
- **Web/Mobile Deployment:** Expand the current Streamlit web interface into a full-stack React or Flutter web application, connected to a Flask/FastAPI backend, to allow on-the-go diagnosis via smartphones for clinical practitioners.
- **Hyperparameter Tuning:** Conduct extensive grid searches for learning rates and custom feature extraction convolution layers.

---

### 7. References
1. Rajpurkar, P. et al. (2018). *MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition* (ResNet).
3. Chollet, F. (2015). *Keras*. GitHub. https://github.com/fchollet/keras
4. TensorFlow Documentation. https://www.tensorflow.org/
