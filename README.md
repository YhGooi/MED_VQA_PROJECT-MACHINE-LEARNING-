# Medical Visual Question Answering (Med-VQA) on PathVQA Dataset

A comprehensive implementation comparing two deep learning approaches for Medical Visual Question Answering on pathology images using the PathVQA dataset.

## 🔬 Project Overview

This project implements and evaluates two distinct architectures for answering questions about medical pathology images:

1. **CNN-Based Baseline**: ResNet-50 + LSTM encoder-decoder
2. **Vision-Language Model (VLM)**: Vision Transformer (ViT) + BERT with Cross-Attention

The models are trained and evaluated on the PathVQA dataset, which contains pathology images with associated questions requiring both closed-ended (yes/no) and open-ended answers.

## 📊 Dataset

**PathVQA Dataset** (`flaviagiammarino/path-vqa` from Hugging Face)
- **Training Set**: 19,755 samples
- **Validation Set**: 6,279 samples
- **Test Set**: 6,761 samples
- **Question Types**: 
  - Closed-ended (yes/no answers)
  - Open-ended (descriptive answers)

## 🏗️ Architecture

### Model 1: CNN-Based Baseline

```
┌─────────────────┐         ┌──────────────────┐
│  Image Input    │         │  Question Input  │
│  (224×224×3)    │         │  (Text)          │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         ▼                           ▼
  ┌─────────────┐            ┌──────────────┐
  │  ResNet-50  │            │  Embedding   │
  │  (Frozen)   │            │      +       │
  │  Features   │            │  Bi-LSTM     │
  └──────┬──────┘            └──────┬───────┘
         │                          │
         │     ┌───────────────┐    │
         └────►│  Concatenate  │◄───┘
               └───────┬───────┘
                       ▼
               ┌──────────────┐
               │  Fusion FC   │
               └──────┬───────┘
                      ▼
               ┌──────────────┐
               │ Classifier   │
               └──────────────┘
```

**Key Features:**
- Pretrained ResNet-50 (ImageNet) with frozen backbone
- Bidirectional LSTM for question encoding
- Concatenation-based fusion
- Classification head for answer prediction
- Optimized for memory efficiency with frozen encoder

### Model 2: Vision-Language Model (VLM)

```
┌─────────────────┐         ┌──────────────────┐
│  Image Input    │         │  Question Input  │
│  (224×224×3)    │         │  (Text)          │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         ▼                           ▼
  ┌─────────────┐            ┌──────────────┐
  │     ViT     │            │     BERT     │
  │  (Frozen)   │            │  (Frozen)    │
  │  Encoder    │            │  Encoder     │
  └──────┬──────┘            └──────┬───────┘
         │                          │
         │     ┌───────────────┐    │
         └────►│ Cross-Attn    │◄───┘
               │ Transformer   │
               │  (Trainable)  │
               └───────┬───────┘
                       ▼
               ┌──────────────┐
               │  Pooling +   │
               │  Classifier  │
               └──────────────┘
```

**Key Features:**
- Pretrained Vision Transformer (ViT) for visual encoding
- Pretrained BERT for text encoding
- Cross-attention mechanism for multimodal fusion
- Only cross-attention and classifier layers are trainable
- Superior semantic understanding through transformer architecture

## 📁 Project Structure

```
MED_VQA_PROJECT-MACHINE-LEARNING-/
├── med_vqa_project.ipynb      # Main Jupyter notebook with full implementation
├── datasets_pathvqa.py        # Custom PyTorch Dataset classes
├── results.json               # Final evaluation metrics
├── README.md                  # Project documentation
├── cnn_baseline_best.pt       # Best CNN model checkpoint (generated)
├── vlm_best.pt               # Best VLM model checkpoint (generated)
├── training_curves.png        # Training visualization (generated)
├── model_comparison.png       # Model comparison plots (generated)
└── results_comparison.png     # Results bar chart (generated)
```

## 🚀 Installation & Setup

### Requirements

```bash
pip install datasets transformers torch torchvision pillow matplotlib scikit-learn tqdm
```

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Torchvision
- Hugging Face Datasets
- PIL (Pillow)
- Matplotlib
- Scikit-learn
- tqdm
- NumPy
- Pandas

## 💻 Usage

### Running the Full Pipeline

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook med_vqa_project.ipynb
   ```

2. **Execute cells sequentially** following the notebook structure:
   - Setup and Imports
   - Dataset Loading
   - Data Preprocessing
   - Model Definitions
   - Training
   - Evaluation
   - Results Visualization

### Dataset Loading

The dataset is automatically loaded from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("flaviagiammarino/path-vqa")
```

### Custom Dataset Class

The `datasets_pathvqa.py` file contains the `PathVQADataset` class with **lazy loading** for memory efficiency:

```python
from datasets_pathvqa import PathVQADataset

train_dataset = PathVQADataset(
    metadata=data_splits['train'],
    hf_dataset=dataset['train'],
    question_vocab=question_vocab,
    answer_vocab=answer_vocab,
    text_preprocessor=text_preprocessor,
    transform=train_transform,
    max_question_len=20,
    max_answer_len=10
)
```

## 🎯 Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 300 (CNN), 300 (VLM) |
| Number of Epochs | 30 |
| Learning Rate (CNN) | 1e-4 |
| Learning Rate (VLM) | 5e-4 |
| Weight Decay | 0.01 |
| Image Size | 224×224 |
| Max Question Length | 20 (CNN), 64 (VLM) |
| Max Answer Length | 10 (CNN), 32 (VLM) |
| Early Stopping Patience | 5 epochs |

### Optimization Techniques

- **Mixed Precision Training (AMP)**: Enabled for GPU acceleration
- **Gradient Clipping**: Max norm = 1.0
- **Learning Rate Scheduling**: 
  - CNN: ReduceLROnPlateau
  - VLM: CosineAnnealingLR
- **Data Augmentation**: Random crops, flips, rotation, color jitter
- **Multi-worker DataLoader**: 10 workers with persistent workers
- **Frozen Encoders**: Both models freeze pretrained backbones for efficiency

## 📈 Results

### Model Performance Comparison

| Metric | CNN Baseline | VLM (ViT+BERT) |
|--------|--------------|----------------|
| **Overall Accuracy** | 0.2701 | **0.5871** |
| **Closed-ended Accuracy** | 0.5399 | **0.8546** |
| **Open-ended F1 Score** | 0.0012 | **0.3534** |
| **Open-ended Precision** | 0.0021 | **0.3601** |
| **Open-ended Recall** | 0.0011 | **0.3557** |

### Key Findings

✅ **VLM Significantly Outperforms CNN Baseline**:
- **2.17× better** overall accuracy
- **1.58× better** on closed-ended questions
- **295× better** F1 score on open-ended questions

✅ **Closed-ended vs Open-ended Performance**:
- Both models perform better on closed-ended (yes/no) questions
- VLM shows much stronger capability for open-ended descriptive answers
- Cross-attention mechanism provides superior multimodal reasoning

✅ **Architecture Insights**:
- Transformer-based architectures excel at medical VQA tasks
- Pretrained vision-language models capture semantic relationships better
- Cross-attention fusion is more effective than simple concatenation

## 🔍 Evaluation Metrics

### Closed-ended Questions
- **Accuracy**: Exact match between predicted and ground truth answers

### Open-ended Questions
- **Token-level F1 Score**: Harmonic mean of precision and recall at token level
- **Precision**: Fraction of predicted tokens that match ground truth
- **Recall**: Fraction of ground truth tokens captured by prediction

## 📊 Visualizations

The project generates three visualization files:

1. **training_curves.png**: Loss and accuracy curves for both models across epochs
2. **model_comparison.png**: Side-by-side comparison of validation metrics
3. **results_comparison.png**: Bar chart of final test set performance

## 🔧 Implementation Details

### Memory Optimization

- **Lazy Loading**: Images loaded on-demand during training
- **Frozen Encoders**: Pretrained backbones frozen to reduce memory footprint
- **Mixed Precision**: FP16 training with automatic mixed precision (AMP)
- **Gradient Checkpointing**: Available for even lower memory usage
- **Efficient Batching**: Optimized batch sizes for GPU memory

### Text Preprocessing

```python
class TextPreprocessor:
    - Lowercase conversion
    - Punctuation removal
    - Whitespace normalization
```

### Vocabulary Building

- Custom vocabulary for CNN baseline with special tokens: `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>`
- BERT tokenizer for VLM with WordPiece tokenization
- Minimum frequency threshold for rare word filtering

## 🎓 Technical Highlights

### Cross-Attention Mechanism

The VLM uses cross-attention to allow question tokens to attend to image patches:

```python
class CrossAttentionBlock(nn.Module):
    - Multi-head cross-attention (8 heads)
    - Layer normalization
    - Feed-forward network (FFN)
    - Residual connections
```

### Training Stability

- Gradient clipping to prevent exploding gradients
- Mixed precision with dynamic loss scaling
- Early stopping to prevent overfitting
- Learning rate scheduling for convergence

## 🚧 Future Improvements

1. **Medical Domain Pretraining**:
   - Use PubMedBERT for better medical text understanding
   - Use medical image pretrained ViT models
   
2. **Advanced Fusion Mechanisms**:
   - Implement LXMERT-style architecture
   - Try UNITER or ViLT models
   
3. **Interpretability**:
   - Add attention visualization
   - Generate explanation for predictions
   
4. **Data Augmentation**:
   - Medical-specific augmentations
   - Mixup and CutMix strategies
   
5. **Ensemble Methods**:
   - Combine CNN and VLM predictions
   - Multi-scale feature fusion


## 📝 License

This project is for educational and research purposes.

## 👥 Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This project was developed as a machine learning research project for medical visual question answering. Results demonstrate the effectiveness of modern vision-language models for healthcare AI applications.