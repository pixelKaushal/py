from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
for k in [1,3,5,7,9,11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    guesses = model.predict(X_test) 
    score = accuracy_score(y_test, guesses)
    print(f"k={k} Accuracy: {score:.2f}")