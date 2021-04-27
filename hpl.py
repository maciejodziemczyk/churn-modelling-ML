#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


# In[2]:


# funkcja do określenia liczby subplotów na podstawie wektora niepustych wykresów
# funkcja zwraca obiekt typu lista [liczba wierszy, liczba kolumn].
# argumenty, dokumentacja:
#     arr - wektor, liczba niepustych wykresów (np. kolumny/wiersze z DataFrame)
#     num_of_cols - int, liczba kolumn, jaką ma mieć nasz subplot

def subplotShape(arr, num_of_cols = 4):
    if num_of_cols > arr.shape[0]:
        raise 'num_of_cols > arr'
    elif arr.shape[0] % num_of_cols != 0:
        for i in range(1, arr.shape[0]): # szukanie reszty z dzielenia
            if arr.shape[0] % num_of_cols == i:
                arr = np.append(arr, np.zeros((1, num_of_cols-i))) # dopełnienie do dzielenia bez reszty w kształcie
                arr = np.reshape(arr, (-1, num_of_cols)) # dopasowanie liczby wierszy do liczby kolumn
                return [arr.shape[0], arr.shape[1]] # lista [liczba wierszy, liczba kolumn]
                break
    else:
        arr = np.reshape(arr, (-1, num_of_cols)) # dopasowanie liczby wierszy do liczby kolumn
        return [arr.shape[0], arr.shape[1]] # lista [liczba wierszy, liczba kolumn]


# In[3]:


def popMeanTest(a, b, alfa = 0.05, alternative = 'two-sided'):
    # import rozkładu normalnego
    from scipy.stats import norm
    
    # statystyka testowa
    stat = (a.mean()-b.mean())/(((a.var()/a.shape[0])+(b.var()/b.shape[0]))**(1/2))
    
    # wartość krytyczna i p_value dla wybranej hipotezy alternatywnej
    if alternative == 'two-sided':
        critical_value = norm.ppf(q = 1-alfa/2)
        p = 2*(1-norm.cdf(abs(stat)))  
    elif alternative == 'one-sided':
        critical_value = norm.ppf(q = 1-alfa)
        p = 1-norm.cdf(abs(stat))
    else:
        raise 'niepoprawny wynik, sprawdź argumenty'
    
    # zwrócenie wyników
    return stat, critical_value, p


# In[4]:


def plotROCsPRs(results):
    
    f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (17, 7))
    
    for true, pred, label in results:
        # Obliczenie punktów potrzebnych do narysowania krzywych ROC i PR
        # funkcja roc_curve zwarca trzy serie danych, fpr, tpr oraz poziomy progów odcięcia
        fpr, tpr, thresholds = roc_curve(true, pred)
        precision, recall, thresholds = precision_recall_curve(true, pred)
        # Obliczamy pole powierzchni pod krzywymi
        rocScore = round(roc_auc_score(true, pred), 4)
        average_precision = round(average_precision_score(true, pred), 4)
        
        # Grubość krzywych
        lw = 2
        
        # Rysujemy krzywą ROC i PR
        ax[0].plot(fpr, tpr, lw = lw, label = '{}: {}'.format(label, rocScore))
        ax[1].plot(recall, precision, lw=lw, label='{}: {}'.format(label, average_precision))
    
    # Rysujemy krzywą 45 stopni jako punkt odniesienia
    ax[0].plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
    
    
    ax[0].set_xlim([-0.01, 1.0])
    ax[0].set_ylim([0.0, 1.01])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver operating characteristic')
    ax[0].legend(loc = "lower right")
    
    ax[1].set_xlim([-0.01, 1.0])
    ax[1].set_ylim([0.0, 1.01])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision Recall Curve')
    ax[1].legend(loc = "lower right")
    
    plt.show()


# In[5]:


def smart_log_var(df, feature):
    '''Funkcja, która policzy logarytm, zamieniając wczesniej wartości mniejsze od zera na 0'''
    df['{}_log'.format(feature)] = df[feature].copy()
    df.loc[df['{}_log'.format(feature)] < 0, '{}_log'.format(feature)] = 0
    df['{}_log'.format(feature)] = np.log(df['{}_log'.format(feature)]+1)


# In[6]:


def linktest(modres, y, data, model = 'logit'):
    
    try:
        # definiowanie wartości dopasowanych i ich kwadratów
        y = pd.Series(np.array(data[y]))
        y_fit = pd.Series(modres.predict())
        yhat = np.log(y_fit/(1-y_fit))
        yhat2 = yhat.copy()**2
        data = pd.concat((y, yhat, yhat2), axis = 1)
        
        # model pomocniczy
        if model == 'logit':
            from statsmodels.formula.api import logit
            reg = smf.glm(formula = 'y~yhat+yhat2', data = df, family = sm.families.Binomial(
                link = sm.genmod.families.links.logit())).fit(disp = False)
        elif model == 'probit':
            from statsmodels.formula.api import probit
            reg = smf.glm(formula = 'y~yhat+yhat2', data = df, family = sm.families.Binomial(
                link = sm.genmod.families.links.probit())).fit(disp = False)
        else:
            raise 'link test only for "logit" and "probit" models'
        
        # zwrócenie wyników
        return reg
            
    except:
        raise 'y should be depended value from model'   


# In[7]:


def unitarization(series):
    maximum = series.max()
    minimum = series.min()
    y = series.copy()
    return y.apply(lambda x: ((x-minimum)/(maximum-minimum)))

