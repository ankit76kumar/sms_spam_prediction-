## 📧 SMS Spam Classifier Using Machine Learning

### 🔍 Overview

This project is a **machine learning-based web application** that classifies SMS or email messages as **Spam** or **Not Spam**. Built using **Streamlit** for deployment and **Natural Language Processing (NLP)** for message preprocessing, it helps identify unsolicited or harmful messages efficiently.

---

### 🎯 Objective

The primary objective of this project is to:

* Automatically detect spam messages using machine learning.
* Help users avoid phishing or promotional content.
* Demonstrate an end-to-end ML pipeline from preprocessing to model deployment.

---

### 💡 Motivation

Spam messages are not only annoying but also a potential security risk. Manual filtering is inefficient, especially at scale. This classifier uses data-driven techniques to automate spam detection, reducing human workload and improving inbox safety.

---

### 🧠 Background

* **Dataset**: A labeled dataset of SMS messages (spam or ham).
* **Tech Stack**:

  * Python
  * Streamlit
  * NLTK (Natural Language Toolkit)
  * Scikit-learn
  * Pickle (for saving models)
* **Model Used**: Likely Multinomial Naive Bayes or Logistic Regression (based on typical use with TF-IDF in spam detection)

---

### 🧱 Project Structure

```
sms-spam-classifier/
│
├── sms-spam-detection.ipynb     # Notebook containing model training & evaluation
├── app.py                       # Streamlit app for deployment
├── vectorizer.pkl               # Saved TF-IDF vectorizer
├── model.pkl                    # Trained machine learning model
├── README.md                    # Documentation (this file)
```

---

### 🧪 Features

* **Text preprocessing**:

  * Lowercasing
  * Tokenization
  * Removing stopwords and punctuation
  * Stemming using Porter Stemmer
* **TF-IDF Vectorization**
* **Spam Classification**
* **Web App UI** using Streamlit

---

### 🖥️ Usage

#### 1. 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

#### 2. 🧠 Predict a Message

* Enter any message into the text box.
* Click on "Predict".
* The app will display: **Spam** or **Not Spam**.

---

### 🧮 How It Works

#### `transform_text(text)`

Performs the following:

* Converts text to lowercase
* Tokenizes the text
* Removes non-alphanumeric characters
* Filters stopwords and punctuation
* Applies stemming

#### Streamlit Logic (`app.py`)

```python
transformed_sms = transform_text(input_sms)
vector_input = tfidf.transform([transformed_sms])
result = model.predict(vector_input)[0]
```

---

### ✅ Example

**Input**:

```
Congratulations! You've won a free ticket to Bahamas. Text WIN to 88888.
```

**Output**:

```
Spam
```

---

### 📌 Dependencies

* `streamlit`
* `nltk`
* `sklearn`
* `pickle`

---

### 🚀 Future Enhancements

* Include confidence scores (e.g., 95% Spam).
* Train with deep learning models (e.g., LSTM).
* Add an API endpoint for backend integration.
* Include SMS source validation to combat phishing.

---

### 📜 License

This project is open-source and free to use for educational and non-commercial purposes.
