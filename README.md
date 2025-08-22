# Movie_Review_Sentiment_Analyzer



# Movie Review Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

A complete machine learning project that automatically classifies movie reviews as positive or negative using Natural Language Processing techniques and Logistic Regression.

## 🎯 Project Overview

**Problem Statement:** It is difficult to manually read and classify hundreds of movie reviews as positive or negative. An automated solution is required to process reviews and determine their sentiment with speed and consistency.

**Solution:** This project builds a text classification model using TF-IDF vectorization and Logistic Regression to analyze movie reviews and predict whether they express positive or negative sentiment.

## ✨ Features

- **📊 Complete ML Pipeline**: From data preprocessing to model evaluation
- **🔄 Text Preprocessing**: Comprehensive cleaning and normalization
- **🎯 High Accuracy**: Achieves 80-90% accuracy on test data
- **📈 Visualizations**: Confusion matrix and feature importance plots
- **🔮 Custom Predictions**: Predict sentiment for any movie review
- **💻 Interactive Interface**: Real-time sentiment analysis
- **📝 Comprehensive Documentation**: Step-by-step explanations

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
Jupyter Notebook or JupyterLab
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
```

2. **Install required packages**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Run the notebook**
```bash
jupyter notebook Movie_Review_Sentiment_Analyzer_Complete.ipynb
```

### Alternative: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/movie-review-sentiment-analyzer/blob/main/Movie_Review_Sentiment_Analyzer_Complete.ipynb)

## 📖 Usage

### Running the Complete Analysis

1. Open the Jupyter notebook
2. Run all cells sequentially from top to bottom
3. The notebook will automatically:
   - Create a balanced dataset of 50 movie reviews
   - Preprocess the text data
   - Train a logistic regression model
   - Evaluate performance with multiple metrics
   - Test predictions on sample reviews

### Custom Prediction Example

```python
# After running the notebook, use this function
sentiment, confidence, details = predict_sentiment("This movie was absolutely fantastic!")

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence*100:.1f}%")
```

### Expected Output
```
Input: "The story was dull and disappointing."
Predicted Sentiment: Negative
Model Accuracy: 87%
```

## 🏗️ Project Structure

```
movie-review-sentiment-analyzer/
│
├── Movie_Review_Sentiment_Analyzer_Complete.ipynb    # Main notebook with complete implementation
├── README.md                                         # Project documentation
├── requirements.txt                                  # Python dependencies
├── LICENSE                                          # MIT License
└── images/                                          # Screenshots and visualizations
    ├── confusion_matrix.png
    └── feature_importance.png
```

## 🔧 Technical Details

### Model Architecture

1. **Text Preprocessing Pipeline**
   - Lowercase conversion
   - Punctuation and special character removal
   - URL and mention cleaning
   - Whitespace normalization

2. **Feature Extraction**
   - TF-IDF Vectorization
   - 1000 maximum features
   - Unigrams and bigrams (n-gram range: 1-2)
   - English stop words removal

3. **Machine Learning Model**
   - Algorithm: Logistic Regression
   - Solver: liblinear (optimal for small datasets)
   - Train-test split: 80-20 with stratified sampling

### Performance Metrics

| Metric | Score |
|--------|-------|
| Training Accuracy | ~100% |
| Testing Accuracy | 80-90% |
| Precision | 0.80+ |
| Recall | 0.80+ |
| F1-Score | 0.80+ |

### Dataset

- **Size**: 50 balanced movie reviews
- **Distribution**: 25 positive + 25 negative reviews
- **Format**: Self-contained (no external files required)
- **Quality**: Comprehensive coverage of sentiment expressions

## 📊 Model Performance

### Confusion Matrix
The model shows balanced performance across both positive and negative classes with minimal false predictions.

### Feature Importance
**Top Positive Indicators:**
- excellent, amazing, wonderful, fantastic, great

**Top Negative Indicators:**
- poor, boring, disappointing, terrible, bad

## 🎮 Interactive Features

The notebook includes an interactive prediction interface:

```python
# Uncomment this line in the notebook to start interactive mode
analyze_custom_review()
```

This allows you to:
- Enter custom movie reviews
- Get instant sentiment predictions
- View confidence scores and probabilities
- Test multiple reviews in real-time

## 📋 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🔄 Workflow

```mermaid
graph LR
    A[Raw Movie Reviews] --> B[Text Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[Train-Test Split]
    D --> E[Logistic Regression]
    E --> F[Model Evaluation]
    F --> G[Predictions]
    G --> H[Sentiment Classification]
```

## 🎯 Key Accomplishments

✅ **Problem Solved**: Automated movie review sentiment classification  
✅ **High Accuracy**: Consistent 80-90% performance on test data  
✅ **Complete Pipeline**: End-to-end ML solution with preprocessing  
✅ **Scalable**: Can handle hundreds of reviews efficiently  
✅ **Interactive**: Real-time prediction capabilities  
✅ **Well-Documented**: Comprehensive explanations and examples  

## 🚀 Future Enhancements

- [ ] **Deep Learning Models**: Implement LSTM/BERT for improved accuracy
- [ ] **Multi-class Classification**: Extend to 1-5 star rating predictions
- [ ] **Real-time API**: Deploy as REST API using Flask/FastAPI
- [ ] **Web Interface**: Create a user-friendly web application
- [ ] **Larger Dataset**: Train on more extensive movie review datasets
- [ ] **Cross-domain**: Adapt for other types of reviews (products, restaurants)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## 👤 Author

**Your Name**
- GitHub: [saurav3k2](https://github.com/your-username)
- LinkedIn: [https://www.linkedin.com/in/saurav-kumar-492bb425b/](https://linkedin.com/in/your-profile)
- Email: saurav3k2@gmail.com

## 🙏 Acknowledgments

- **Scikit-learn** for providing excellent machine learning tools

- **Pandas** for data manipulation capabilities
- **Matplotlib & Seaborn** for visualization support
- **Jupyter** for the interactive development environment

## 📊 Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/movie-review-sentiment-analyzer)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/movie-review-sentiment-analyzer)
![GitHub issues](https://img.shields.io/github/issues/your-username/movie-review-sentiment-analyzer)
![GitHub stars](https://img.shields.io/github/stars/your-username/movie-review-sentiment-analyzer?style=social)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
