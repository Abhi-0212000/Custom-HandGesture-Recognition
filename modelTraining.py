import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

randomForestClassifier = RandomForestClassifier()


randomForestClassifier.fit(x_train, y_train)



y_predict = randomForestClassifier.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly'.format(score*100))

# to save the best model
f = open('model.p', 'wb')
pickle.dump({'model':randomForestClassifier}, f)
f.close()