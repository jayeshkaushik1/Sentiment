#Sentiment Analysis Project
This repository hosts a Jupyter Notebook dedicated to sentiment analysis. The project aims to classify text into positive, negative, or neutral sentiments by employing both classical machine learning techniques with TF-IDF vectorization and advanced deep learning methods leveraging BERT embeddings.

##🚀 Project Overview
This project outlines a comprehensive sentiment analysis pipeline. It evaluates traditional machine learning models using TF-IDF features and integrates the Hugging Face Transformers library to utilize pre-trained BERT models for sophisticated text representation. The core objective is to benchmark the performance of these diverse approaches in sentiment classification tasks.

##✨ Features
Data Handling: Seamless loading of train.csv and test.csv datasets.
Robust Null Value Management: Custom function for intelligent handling of missing values based on their percentage in the dataset.
Comprehensive Text Preprocessing: Includes functionalities for:
Lowercasing text
Removal of custom English stopwords (preserving negations)
Stripping URLs
Eliminating punctuation
Normalizing (removing) extra spaces
Insightful Exploratory Data Analysis (EDA): Visualizations covering:
Distribution of tweet lengths per sentiment category.
Sentiment label distribution across tweet times and user age groups.
Identification of most frequent words in positive, negative, and neutral tweets.
TF-IDF Vectorization: Transforms raw text data into numerical feature vectors using TfidfVectorizer.
Classical Machine Learning Models: Implementation and evaluation of:
Logistic Regression
Multinomial Naive Bayes
Linear SVM
Random Forest
Performance Metrics: Provides detailed accuracy scores, confusion matrices, and classification reports for classical ML models.
BERT Embeddings Integration: PyTorch Dataset classes (TextDatasetWithEmbeddings, TokenizedDataset) are included for:
Preparing data for pre-trained BERT models.
Pre-computing BERT embeddings in batches for optimized GPU memory utilization.
##⚙️ Installation
To set up the project locally, follow these steps:

Clone the repository:
git clone https://github.com/jayeshkaushik1/Sentiment.git
cd Sentiment
Install dependencies:
pip install pandas numpy nltk scikit-learn matplotlib seaborn torch transformers
Download NLTK stopwords:
import nltk
nltk.download('stopwords')
##📊 Dataset
The project relies on two CSV files: train.csv and test.csv. These datasets should be placed in the project's root directory. The notebook verifies their presence at runtime.

The train_df is expected to contain columns such as textID, text, selected_text, sentiment, Time of Tweet, Age of User, Country, Population -2020, Land Area (Km²), and Density (P/Km²).

##🧹 Data Preprocessing
The preprocess_text function executes a series of text cleaning operations: lowercasing, removal of stopwords (with negation retention), URL stripping, punctuation removal, and extra space normalization.

Missing values are managed by the handle_null_values function. Rows with less than 5% missing values are removed; otherwise, rows are retained based on crucial columns like "text" and "sentiment".

##🧠 Models
Classical Machine Learning Models
TfidfVectorizer is applied to transform processed text into numerical features, which are then used to train and evaluate the following models:

Logistic Regression
Multinomial Naive Bayes
Linear SVM
Random Forest
Deep Learning Data Preparation
For deep learning methodologies, BertTokenizer and BertModel are utilized to generate embeddings. The TextDatasetWithEmbeddings class handles the batch pre-computation of BERT embeddings to optimize memory, while TokenizedDataset prepares raw tokenized inputs for model consumption.

##📈 Results
Initial performance of the classical machine learning models on the test set:

Logistic Regression: 70.49%
Naive Bayes: 63.38%
Linear SVM: 69.16%
Random Forest: 64.88%
The notebook also provides *confusion matrices* and comprehensive classification reports for both test and validation datasets. Visualizations of word frequencies and sentiment distributions offer deeper insights into the dataset's characteristics.

##🚀 Usage
To execute the sentiment analysis:

Ensure train.csv and test.csv are in your project's root.
Open the Sentiment (2) (2).ipynb notebook in a Jupyter environment (e.g., Google Colab, JupyterLab).
Run all cells sequentially to initiate data loading, preprocessing, model training, and evaluation.
##🤝 Contributing
Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests for any enhancements or bug fixes.

##📄 License
This project is licensed under the MIT License.


