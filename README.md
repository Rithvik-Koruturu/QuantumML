# Quantum-ML-lab-Rithvik
## Quantum Algorithms & Quantum Machine Learning Implementations in PennyLane

This repository contains Python implementations of popular quantum algorithms and quantum machine learning (QML) techniques using **PennyLane**, a quantum computing library.

---

### Implemented Quantum Algorithms:
- **Deutsch's Algorithm**
- **Deutsch-Josza Algorithm**
- **Bernstein-Vazirani Algorithm**

---

###  Quantum Machine Learning Additions:
New commits have introduced quantum image encoding techniques for training QML models. These include:

1. **Quantum Image Encoding with Amplitude Encoding**
   - Encodes classical images into quantum states using amplitude encoding.
   - Efficiently represents images for quantum processing.

2. **Feature Extraction and PCA-based Encoding**
   - Extracts essential features from classical images.
   - Uses Principal Component Analysis (PCA) to reduce dimensionality before encoding into quantum states.

3. **Quantum Feature Encoding using VGG16 and Amplitude Embedding**
   - Leverages VGG16 for feature extraction.
   - Converts extracted features into quantum states using amplitude embedding.

4. **V4.X – Quantum Image Preprocessing using ORB & PCA with Amplitude Embedding**
   - ORB descriptors are extracted from images for keypoint-based feature extraction.
   - Dimensionality reduced using PCA.
   - Features are amplitude-encoded into quantum states.
   - Suitable for building lightweight QML pipelines using hybrid features.

5. **V5 – Hybrid Quantum Convolutional Neural Network (QCNN) with QAOA and Amplitude Feature Embedding**
   - Combines classical feature extraction with quantum embedding and QAOA circuits.
   - Uses amplitude encoding after dimensionality reduction for input features.
   - Trains QCNN models using PennyLane for image classification tasks (e.g., Cats vs Dogs).
   - Supports flexible quantum circuit design with tunable layers and depth.

---


## 🗂 Architecture & File Structure
```

Quantum-ML-lab-Rithvik/
│
├── README.md
│
├── Encode_and_Preprocess_Images/
│   ├── V1-Encode and Preprocess.ipynb
│   ├── V1.Encode_and_preprocess_documentation.pdf
│   ├── V2-Encode and Preprocess.ipynb
│   ├── V2.Encode_and_preprocess_documentation.pdf
│   ├── V3-Encode and Preprocess_VGG16.ipynb
│   ├── V3.Encode_and_preprocess_documentation.pdf
│   ├── V4-Encode and Preprocess.ipynb
│   ├── V4.Encode_and_preprocess_documentation.pdf
│   ├── V5-Encode and Preprocess.ipynb
│   ├── V5.Encode_and_preprocess_documentation.pdf
│   └── comparision_encode and preprocess.pdf
│
├── Quantum Algorithms/
│   ├── Bernstein-Vazirani Algorithm empl1.ipynb
│   ├── Bernstein-Vazirani Algorithm empl2.ipynb
│   ├── Deutchs - Jozsa ALgorithm.ipynb
│   └── Deutchs Algorithm_RK.ipynb
│
└── Quantum Neural Networks/
    ├── Hybrid_QNN_Binary_Image_Classification.ipynb
    └── QNN Binary Classification - iris dataset.ipynb
```


## 📦 Prerequisites
To run the code, install the following libraries:

- [PennyLane](https://pennylane.ai/)
- [NumPy](https://numpy.org/)
- [TensorFlow](https://www.tensorflow.org/) (for VGG16-based encoding)
- [Scikit-learn](https://scikit-learn.org/) (for PCA-based encoding)
- [OpenCV](https://opencv.org/) (for ORB keypoint detection)

---

##  Explanation of Structure:

### Quantum Algorithm Implementations:
Each algorithm (Deutsch's, Deutsch-Josza, Bernstein-Vazirani, Simon's) has its own section. This includes:
- **Cell 1**: Defining the quantum device.
- **Cell 2**: Defining the oracle.
- **Cell 3**: Writing the quantum algorithm.
- **Cell 4**: Printing the measurement results.
- **Cell 5**: Drawing the quantum circuit using `draw_mpl`.

### Quantum Image Encoding Modules:
Each encoding technique has its own module, ensuring clear separation and easy integration into QML models.

### Code Snippets:
- Code follows a structured format with comments for each phase of the quantum/classical hybrid process.

---

## 🚀 Contribution & Future Work:
- Add quantum GANs and VQAs for generative and discriminative learning.
- Introduce multi-class classification pipelines using QCNNs.
- Explore quantum-enhanced attention mechanisms.

---

**Crafted with ♥ by Rithvik Koruturu**
