# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam dataset and handle encoding properly.
2. Display basic information and check for null values.
3. Extract the message text as features (x) and labels (y) for classification.
4. Split the dataset into training and testing sets.
5. Convert the text data into numerical vectors using CountVectorizer.
6. Train an SVM classifier on the transformed training data.
7. Predict on test data and evaluate model accuracy using accuracy_score.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Elavarasan M
RegisterNumber:  212224040083
*/
```
```
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
```
```
data = pd.read_csv("spam.csv", encoding="Windows-1252")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
# separating the features and labels
x = data["v2"].values  # text messages
y = data["v1"].values  # labels: spam or ham
```
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
```
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
```
```
svc = SVC()
svc.fit(x_train, y_train)
```
```
y_pred = svc.predict(x_test)
y_pred
```
```
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
```

## Output:
**Head Values**

![Screenshot 2025-05-12 184142](https://github.com/user-attachments/assets/2c07e202-4706-418d-9d19-e5224d28abeb)

**Dataframe Info**

![Screenshot 2025-05-12 184148](https://github.com/user-attachments/assets/7a4e67f2-470d-4f07-a8a5-2bdede07a21a)

**Sum - Null Values**

![Screenshot 2025-05-12 184153](https://github.com/user-attachments/assets/503f25a5-2c20-4d26-9d5f-6be76bf7ebf8)

**Training the model**

![Screenshot 2025-05-12 184200](https://github.com/user-attachments/assets/fe479bf2-d626-4072-8150-61c868cfbcc7)

**Predicting the test data**

![Screenshot 2025-05-12 184206](https://github.com/user-attachments/assets/22cfa7ad-806c-48e8-a889-fa8fae6ddcad)

**Accuracy**

![Screenshot 2025-05-12 184214](https://github.com/user-attachments/assets/f784c53c-f2dd-41ce-a79f-04170086a892)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
