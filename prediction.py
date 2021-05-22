import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine


TEST_SIZE = 0.003

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=4)

    model.fit(evidence, labels)
    return model

def load_data(file_name):
    four_letters = []
    final_letters = []
    # opening the text file 
    with open(file_name,'r') as file: 
    
        # reading each line     
        for line in file: 

            # reading each word         
            for word in line.split(): 
                if ("S" not in word):
                    # displaying the words            
                    four_letters.append([to_int(k) for k in word[:4]])
                    final_letters.append( to_int(word[4:])) 
    return (four_letters, final_letters)

def to_int(k):
    k = k.lower()
    
    if k in letters:
        for i in range(len(letters)):
            if k == letters[i]:
                return i
    else:
        print(f"Error: {k}")

def to_letter(k):
    return letters[k]


if len(sys.argv) != 2:
    sys.exit("Usage: python shopping.py data")

evidence, labels = load_data(sys.argv[1])
X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
)

# Train model and make predictions
model = train_model(X_train, y_train)

predictions = model.predict(X_test)


svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_disp = plot_roc_curve(svc, X_test, y_test)

print(len(X_test))
print(len(X_train))

for k in range(len(predictions)):
    first = ""
    for i in X_test[k]:
        first += to_letter(i)
    print(f"{first}{to_letter(predictions[k])}")


while True:
    input_str = input("Four letter word:")
    test_set = []
    test_set.append([to_int(k) for k in input_str[:4]])
    print(test_set)
    predictions = model.predict(test_set)
    print(predictions)
    for k in range(len(predictions)):
        first = ""
        for i in test_set[k]:
            first += to_letter(i)
        print(f"{first}{to_letter(predictions[k])}")
    