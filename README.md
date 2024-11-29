# **Contextual Analysis of Product Reviews using Text Mining and Clustering**

## **Project Overview**

This project aims to analyze Amazon product reviews using various text mining techniques, such as sentiment analysis, topic modeling, and clustering. The goal is to extract meaningful insights from customer feedback, classify reviews based on sentiment (positive, negative, neutral), identify latent topics, and group similar reviews together through clustering. The project also includes visualizations to better understand the results.

## **Features**

- **Sentiment Analysis**: Classifies reviews into positive, negative, or neutral categories.
- **Topic Modeling**: Identifies and displays topics within reviews using NMF and LDA techniques.
- **Clustering**: Groups reviews into clusters based on their content using KMeans clustering.
- **Visualization**: Visualizes sentiment distribution, word frequencies, and clustering results with PCA for dimensionality reduction.
- **Evaluation**: Evaluates the sentiment analysis model using Logistic Regression and performance metrics.

## **Software Requirements**

- Python 3.x
- Google Colab or Jupyter Notebook
- Libraries:
  - `pandas`
  - `nltk`
  - `sklearn`
  - `seaborn`
  - `matplotlib`
  - `transformers`
  - `textblob`

## **Installation**

1. **Google Colab**
   - Open Google Colab and create a new notebook.
   - Install the required libraries by running the following command in the first cell(optional):
     ```bash
     !pip install pandas nltk scikit-learn seaborn matplotlib transformers textblob kaggle
   - You can now start running the code step-by-step present in the 'Contextual_Analysis_of_Product_Reviews.ipynb' notebook

2. **Jupyter Notebook**
   If you prefer to run the project locally in Jupyter Notebook, follow these steps:
   - Open Jupyter Notebook and create a new notebook.
   - Install the required libraries by running the following in a notebook cell(optional):
     ```bash
     !pip install pandas nltk scikit-learn seaborn matplotlib transformers textblob kaggle
   - After setting up, you can begin running the code in the notebook.

Note: The dataset is already downloaded and unzipped in the notebook. So, you can do it directly by running the code. But if you prefer to know the process, the dataset is hosted on Kaggle. To download it, you need to have Kaggle credentials and the kaggle API installed. Follow these steps:
   - Create a Kaggle account
   - Download the Kaggle API key from Kaggle API credentials page.
   - Download the dataset using the following command:
     ```bash
     kaggle datasets download -d datafiniti/consumer-reviews-of-amazon-products
     unzip consumer-reviews-of-amazon-products.zip
     
## **How to Run**

1. After following the installation process, you can directly run the code in your specific notebook.
2. The code will:
   - Load the dataset.
   - Clean and preprocess the review text data.
   - Perform sentiment analysis using both transformer-based models and TextBlob.
   - Apply Word Frequency Analysis
   - Apply topic modeling techniques (NMF and LDA).
   - Apply KMeans clustering to group similar reviews.
   - Visualize the results (sentiment distribution, word frequencies, clustering).
   - Improves the silhouette score and gives better cluster visualization
   - Save the processed and labeled dataset.

## **Results**

1. Sentiment Analysis: The reviews will be classified into positive, negative, and neutral categories.
2. Topic Modeling: The top words associated with five latent topics from the reviews will be displayed.
3. Clustering: Reviews will be grouped into 5 clusters, and these clusters will be visualized using PCA (Principal Component Analysis).
4. Visualizations:
   - A count plot showing the distribution of sentiments (positive, negative, neutral).
   - A bar chart displaying the most common keywords in the reviews.
   - A 2D scatter plot visualizing clusters after applying PCA for dimensionality reduction.

## **Troubleshooting**
1. Issue: Kaggle Dataset Download Fails
   Ensure that the Kaggle API key is correctly set up. You can check your environment variable KAGGLE_CONFIG_DIR to confirm this.
   Ensure you have the kaggle Python package installed (pip install kaggle).
2. Issue: Clustering Results Look Poor
   KMeans clustering might not perform well due to the high dimensionality of the data. Apply PCA or other dimensionality reduction techniques before clustering for better results(which is done in the code already).

## **Contact**
For any questions or suggestions, feel free to contact me via:
- GitHub: sowmya1730
- Email: saisowmya.bandaluppi@gwu.edu
- GitHub Repo: https://github.com/sowmya1730/Contextual-Analysis-of-Product-Reviews-using-Text-Mining-and-Clustering

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.
