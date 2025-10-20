# Sentiment Analysis of Product Reviews 

## 📌 Project Description  
This project focuses on analyzing customer reviews of products and services. The main goal is to analyze how it can automatically determine the **sentiment** of a review (positive, neutral, or negative).

## 🎯 Objectives  
- Collect and preprocess reviews from [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews?resource=download).  
- Explore the relationship between review texts and user ratings.  
- Build **BERT** and **CatBoost** models and determine the best metrics for category scores.  
- Perform clustering of reviews to detect hidden patterns. Give each cluster a name 
- Compare model performance and summarize insights.  
- Identify sarcastic reviews and determine their number.

## 🗂️ Data  
- Source: **Amazon Product Reviews**  
- Format: review texts and numerical user ratings (1–5).  
- Processing: text cleaning, tokenization, lemmatization, mapping ratings to sentiment classes.  

## ⚙️ Methods & Tools  
- **NLP:** BERT, tokenization, embeddings.  
- **ML:** CatBoost, evaluation metrics (Recall, Precision, F1-score).  
- **Clustering:** KMeans / DBSCAN for grouping reviews by themes.  
- **Data Engineering:** data collection, cleaning, preprocessing.  
- **Visualization:** matplotlib, seaborn, plotly.  

## 👥 Team Roles  
- **Renata** — Product Manager (goal setting, stakeholder communication).  
- **Alexander** — Project Manager (planning and timeline control).  
- **Elizaveta** — Data Analyst (EDA, visualization, reporting).  
- **Ivan** — Data Engineer (data collection, cleaning, preparation).  
- **Ksenia** — Data Scientist (modeling, training, optimization).  

## 📈 Expected Outcomes  
- A prepared dataset with labeled sentiments.  
- Comparative analysis of model performance (BERT vs CatBoost).  
- Clustering results showing main themes in reviews.  
- Practical conclusions on model applicability for product/service feedback analysis.  

## 📊 Evaluation Metrics  
- **Accuracy**  
- **Precision, Recall, F1-score**  
- **ROC-AUC** (for binary classification tasks).

# 🚀 How to check results 

Clone the Repository
```bash
git clone https://github.com/Sasmas314/pnot_1.git
cd pnot_1
```


# deploy_sentiment_model
