# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:41:36 2021

@author: ide23
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import matplotlib.pyplot as plt

# sql_connection.py neprikladam kedze su to sukromne data
from sql_connection import engine as engine

# control_select.py neprikladam lebo je to len citanie z excel suboru,
# kde su id potkana a obdobie kedy boli v kontrolnej etape.
# Pomocou toho info selectnute data. 
from control_select import control_df

# Aby sme vedeli priradit podla idpotkana strain
strain_map = pd.read_sql(
        "SELECT `id`, `strain` FROM `ratExperiment`",
        con=engine
    )

# Jednotlivim strainom priradime farbu
# =============================================================================
# colors_mapa = {'SHR': '#F5793A', 'WT': '#A95AA1', 'HanSD': '#0F2080', 'TGR': '#85C0F9'}
# 
# encoder = {'SHR': '0', 'WT': '1', 'HanSD': '2', 'TGR': '3'}
# =============================================================================


def vyrataj_priemery(stlpec):

    pomocna = control_df[['idRatExperiment', stlpec, 'lightIntensity']]
    pivot_rats = pomocna.groupby(['idRatExperiment', 'lightIntensity']).mean()
    pivot_rats = pivot_rats.reset_index(level=[0, 1])
    X = pivot_rats[['idRatExperiment', stlpec]][pivot_rats.lightIntensity == 0]
    Y = pivot_rats[['idRatExperiment', stlpec]][pivot_rats.lightIntensity == 150]
    rats_avgs = X.set_index('idRatExperiment').join(Y.set_index('idRatExperiment'), lsuffix='dark', rsuffix='light')

    return rats_avgs

def vyrataj_priemery_mnoho(*args):
        iterargs = iter(args)
        x = vyrataj_priemery(next(iterargs))
        for stlpec in iterargs:
            x = x.join(vyrataj_priemery(stlpec))

        x = x.join(strain_map.set_index('id'))

        return x

tabulka_priemery = vyrataj_priemery_mnoho('pulsePressure', 'diastolicBP', 'meanBP', 'systolicBP', 'heartRate', 'activity')






################ PRIPRAVA #############
X = np.array(tabulka_priemery[['pulsePressuredark', 'pulsePressurelight',
                               'diastolicBPdark', 'diastolicBPlight',
                               'meanBPdark', 'meanBPlight',
                               'systolicBPdark', 'systolicBPlight',
                               'heartRatedark', 'heartRatelight',
                               'activitydark', 'activitylight']])


y = np.ravel(np.array(tabulka_priemery[['strain']]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

################################## SVM #################################
param_grid = {
    'C': [2**(2*i-1) for i in range(-2,9)],  
    'gamma': [2**(2*i-1) for i in range(-7,3)], 
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
  
clf = GridSearchCV(SVC(), param_grid, scoring='f1_micro', n_jobs = -1)
clf.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % clf.best_score_)
print(clf.best_params_)

clf.best_estimator_.predict(X_train)
pred = clf.best_estimator_.predict(X_test)

print(confusion_matrix(y_test, pred))    
print(classification_report(y_test, pred))


################################ RandomForest ##############################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

label_binar = LabelBinarizer()
y_train_bin = label_binar.fit_transform(y_train)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train_bin)

rf = RandomForestRegressor(**rf_random.best_params_)
rf.fit(X_train, y_train_bin)

print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, y_train_bin),
                                                                                             rf.score(X_test, label_binar.transform(y_test))))
# Tu zistime ze RF nie je az tak dobre na tuto klasifikaciu
pred = label_binar.inverse_transform(rf.predict(X_test))
print(confusion_matrix(y_test, pred))    
print(classification_report(y_test, pred))


random_forest_importance = [(feature, round(importance, 2)) for feature, importance in zip(list(tabulka_priemery.columns[0:12]), list(rf.feature_importances_))]
random_forest_importance = sorted(random_forest_importance, key = lambda x: x[1], reverse = True)

############################## PermutationImportance ########################


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(clf.best_estimator_).fit(X_test, y_test)


perm_importance = [(feature, round(importance, 2)) for feature, importance in zip(list(tabulka_priemery.columns[0:12]), list(perm.feature_importances_))]
perm_importance = sorted(perm_importance, key = lambda x: x[1], reverse = True)



############################### SequentialFeatureSelector ##################
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

estimator = SVC()
estimator.set_params(**clf.best_params_)
sfs1 = sfs(estimator,
           k_features=6,
           verbose=2,
           scoring='accuracy',
           cv=5)

sfs1 = sfs1.fit(X_train, y_train)

tabulka_priemery.columns[list(sfs1.k_feature_idx_)]


############################### Vysledky ###########################
# Podla RF
vyber = [tabulka_priemery.columns.get_loc(stlpec) for stlpec, skore in random_forest_importance]
estimator.fit(X_train[:, vyber[0:6]], y_train)
pred = estimator.predict(X_test[:, vyber[0:6]])

print(confusion_matrix(y_test, pred))    
print(classification_report(y_test, pred))

# Podla Permutacie

vyber = [tabulka_priemery.columns.get_loc(stlpec) for stlpec, skore in perm_importance]
estimator.fit(X_train[:, vyber[0:6]], y_train)
pred = estimator.predict(X_test[:, vyber[0:6]])

print(confusion_matrix(y_test, pred))    
print(classification_report(y_test, pred))

# Podla Sequential
vyber = list(sfs1.k_feature_idx_)
estimator.fit(X_train[:, vyber[0:6]], y_train)
pred = estimator.predict(X_test[:, vyber[0:6]])

print(confusion_matrix(y_test, pred))    
print(classification_report(y_test, pred))

# Podla mna
moj_vyber = ['meanBPdark', 'meanBPlight', 'heartRatedark', 'heartRatelight', 'activitydark', 'activitylight']
vyber = [tabulka_priemery.columns.get_loc(stlpec) for stlpec in moj_vyber]
estimator.fit(X_train[:, vyber[0:6]], y_train)
pred = estimator.predict(X_test[:, vyber[0:6]])

print(confusion_matrix(y_test, pred))    
print(classification_report(y_test, pred))




