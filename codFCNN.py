import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import tensorflow
from sklearn.metrics import make_scorer, confusion_matrix


dataset = pd.read_csv('datasetBinarioTCC.csv')
X = dataset.iloc[:, 0:249].values 
y = dataset.iloc[:, 249].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# Normaliza a base de dados
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def custom_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tn / (tn + fp))

def custom_sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp / (tp + fn))

def build_classifier(optimizer, activation, activationOutput):
    classifier = Sequential()
    classifier.add(Dense(units = 160, activation = activation, input_dim = 249)) 
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 160, activation = activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 160, activation = activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 160, activation = activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 160, activation = activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 160, activation = activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 160, activation = activation))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 1, activation = activationOutput)) 
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [128],
              'epochs': [500],
              'optimizer': ['adam'],
              'activation': ['relu'],
              'activationOutput': ['sigmoid']}

metricas = {'accuracy'      :   'accuracy',
            'auc'           :   'roc_auc',
            'f1'            :   'f1',
            'precision'     :   'precision',
            'recall'        :   'recall',
           'sensitivity'    :   make_scorer(custom_sensitivity),
           'specificity'    :   make_scorer(custom_specificity)}

grid_search = GridSearchCV(estimator = classifier,
                           verbose = 2,
                           param_grid = parameters,
                           n_jobs = None,
                           scoring = metricas,
                           refit = 'accuracy', 
                           return_train_score = False,
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
print(best_parameters)
best_accuracy = grid_search.best_score_

print("Resultado detalhado do experimento")
print(grid_search.cv_results_)
print('Fold', 'Accuracy', 'AUC', 'F1', 'Sensitivity', 'Specificity', 'Precision', 'Recall')
print('0', grid_search.cv_results_['split0_test_accuracy'], grid_search.cv_results_['split0_test_auc'], grid_search.cv_results_['split0_test_f1'], grid_search.cv_results_['split0_test_sensitivity'], grid_search.cv_results_['split0_test_specificity'], grid_search.cv_results_['split0_test_precision'], grid_search.cv_results_['split0_test_recall'])
print('1', grid_search.cv_results_['split1_test_accuracy'], grid_search.cv_results_['split1_test_auc'], grid_search.cv_results_['split1_test_f1'], grid_search.cv_results_['split1_test_sensitivity'], grid_search.cv_results_['split1_test_specificity'], grid_search.cv_results_['split1_test_precision'], grid_search.cv_results_['split1_test_recall'])
print('2', grid_search.cv_results_['split2_test_accuracy'], grid_search.cv_results_['split2_test_auc'], grid_search.cv_results_['split2_test_f1'], grid_search.cv_results_['split2_test_sensitivity'], grid_search.cv_results_['split2_test_specificity'], grid_search.cv_results_['split2_test_precision'], grid_search.cv_results_['split2_test_recall'])
print('3', grid_search.cv_results_['split3_test_accuracy'], grid_search.cv_results_['split3_test_auc'], grid_search.cv_results_['split3_test_f1'], grid_search.cv_results_['split3_test_sensitivity'], grid_search.cv_results_['split3_test_specificity'], grid_search.cv_results_['split3_test_precision'], grid_search.cv_results_['split3_test_recall'])
print('4', grid_search.cv_results_['split4_test_accuracy'], grid_search.cv_results_['split4_test_auc'], grid_search.cv_results_['split4_test_f1'], grid_search.cv_results_['split4_test_sensitivity'], grid_search.cv_results_['split4_test_specificity'], grid_search.cv_results_['split4_test_precision'], grid_search.cv_results_['split4_test_recall'])
print('5', grid_search.cv_results_['split5_test_accuracy'], grid_search.cv_results_['split5_test_auc'], grid_search.cv_results_['split5_test_f1'], grid_search.cv_results_['split5_test_sensitivity'], grid_search.cv_results_['split5_test_specificity'], grid_search.cv_results_['split5_test_precision'], grid_search.cv_results_['split5_test_recall'])
print('6', grid_search.cv_results_['split6_test_accuracy'], grid_search.cv_results_['split6_test_auc'], grid_search.cv_results_['split6_test_f1'], grid_search.cv_results_['split6_test_sensitivity'], grid_search.cv_results_['split6_test_specificity'], grid_search.cv_results_['split6_test_precision'], grid_search.cv_results_['split6_test_recall'])
print('7', grid_search.cv_results_['split7_test_accuracy'], grid_search.cv_results_['split7_test_auc'], grid_search.cv_results_['split7_test_f1'], grid_search.cv_results_['split7_test_sensitivity'], grid_search.cv_results_['split7_test_specificity'], grid_search.cv_results_['split7_test_precision'], grid_search.cv_results_['split7_test_recall'])
print('8', grid_search.cv_results_['split8_test_accuracy'], grid_search.cv_results_['split8_test_auc'], grid_search.cv_results_['split8_test_f1'], grid_search.cv_results_['split8_test_sensitivity'], grid_search.cv_results_['split8_test_specificity'], grid_search.cv_results_['split8_test_precision'], grid_search.cv_results_['split8_test_recall'])
print('9', grid_search.cv_results_['split9_test_accuracy'], grid_search.cv_results_['split9_test_auc'], grid_search.cv_results_['split9_test_f1'], grid_search.cv_results_['split9_test_sensitivity'], grid_search.cv_results_['split9_test_specificity'], grid_search.cv_results_['split9_test_precision'], grid_search.cv_results_['split9_test_recall'])
print('Mean', grid_search.cv_results_['mean_test_accuracy'], grid_search.cv_results_['mean_test_auc'], grid_search.cv_results_['mean_test_f1'], grid_search.cv_results_['mean_test_sensitivity'], grid_search.cv_results_['mean_test_specificity'], grid_search.cv_results_['mean_test_precision'], grid_search.cv_results_['mean_test_recall'])

#print(grid_search.best_estimator_.model.history.history)