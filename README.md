# Python_intership_task5
# 📧 Spam Email Classifier

This project is a simple machine learning model built in Python that classifies emails as **Spam** or **Ham (Not Spam)** using TF-IDF vectorization and Logistic Regression.

---

## 🚀 Features

- Reads and processes email data (`mail_data.csv`)
- Converts email text to numerical features using **TF-IDF**
- Trains a **Logistic Regression** model
- Allows user to input custom email text to classify
- Prints whether it's **Spam** or **Ham**

---

## 🧠 Model Used

- **TF-IDF Vectorizer** for feature extraction
- **Logistic Regression** from `sklearn` for binary classification

---

## 🗂️ Project Structure

Spam_email_classifier.py
mail_data.csv
README.md



---

## 📦 Requirements

Install these using pip:

```bash
pip install numpy pandas scikit-learn

```

---
## 🏃‍♂️ How to Run
1.Make sure mail_data.csv is in the same folder.
2.Run the Python file:
```
python Spam_email_classifier.py
```
3.Enter the email content when prompted.
4.The model will tell if the email is Spam or Ham.

---

## 📊 Dataset Format
mail_data.csv should contain:
| Category | Message              |
| -------- | -------------------- |
| ham      | Hello, how are you?  |
| spam     | Win cash prizes now! |
| ...      | ...                  |

## 🧪 Sample Output
```
Enter mail content: Congratulations! You have won $1000 gift card.
Spam mail

Enter mail content: Meeting rescheduled to 3 PM. Please confirm.
Ham Mail
```
