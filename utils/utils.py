import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from numpy.random import randint
import yaml
from typing import List, Tuple, Any, Union

def read_yaml(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# Functions for analysing dataset
def print_correlation_matrix(correlation: pd.DataFrame, title: str):
  fig, ax = plt.subplots(1,1,figsize=(15,10))
  correlation.style.background_gradient(cmap='Blues').set_precision(2)
  sns.heatmap(correlation, cmap='Blues', ax=ax, annot=True, fmt='.1g')
  ax.set_title(title, fontsize=18, y=1.05)
    
# Data preprocessing functions
def bootstrap(X, bootstrap_size=400):
  # create bootstrap of X of given size
  current_samples = len(X)
  indexes_b = randint(0, current_samples, bootstrap_size)
  X_bootstraps = X.iloc[indexes_b].copy()
  return X_bootstraps

def estimate_mean_stdev_with_bootstrap(X: pd.DataFrame, n_bootstraps: int, bootstrap_size: int=400):
  # create n_bootstraps sets from X and use it to calculate more precise stats
  X_bootstrap_sets = [bootstrap(X, bootstrap_size) for _ in range(n_bootstraps)]
  means = [X_set.mean() for X_set in X_bootstrap_sets]
  stdevs = [X_set.std() for X_set in X_bootstrap_sets]

  est_mean = pd.DataFrame(means).mean()
  est_stdev = pd.DataFrame(stdevs).mean()

  err_mean = (pd.DataFrame([(est_mean-x)**2 for x in means]).sum()/(len(X_bootstrap_sets)-1))**0.5
  err_stds = (pd.DataFrame([(est_stdev-x)**2 for x in stdevs]).sum()/(len(X_bootstrap_sets)-1))**0.5

  return est_mean, est_stdev, err_mean, err_stds

# Plots functions
def plot_histograms(df: pd.DataFrame, columns: List[float], title: str, color: str, figsize: Tuple[int, int]=(12,5), legend: bool=True):
  plt.style.use('ggplot')
  df.hist(column=columns, figsize=figsize, rwidth=0.5, color=color)
  plt.title(title, fontsize=18, y=1.05)
  plt.xlabel(columns[0])
  plt.ylabel('Number of students')
  plt.xticks(np.arange(21), np.arange(21))
  plt.show()

def print_histograms(data: pd.DataFrame, subtitles: Union[str, None]=None, title: str='insert_title', figsize: Tuple[int, int]=(10,5)):
  plt.style.use('ggplot')
  shape = data.shape
  n_isto =  shape[1]
  fig, ax = plt.subplots(1, n_isto, figsize=figsize, sharey=False)
  maxy = []
  #for i in range(n_isto):
  x = data[:,0]
  possible_qualities = np.arange(11) # possible quality from 0 to 11
  values = [len(x[x == i]) for i in possible_qualities] # count the number of wines per quality
  values, _, _ = ax.hist(x, np.arange(12), density=False, rwidth=0.8)
  if subtitles is not None:
    ax.set_title(subtitles, fontsize=14)
  plt.sca(ax)
  ax.set_xlim((-0.5,11.5))
  ax.set_xlabel('Quality')
  ax.set_ylabel('Number of Wines')
  plt.xticks(np.arange(11)+.5, np.arange(11))
  maxy.append(max(values))
  fig.suptitle(title, fontsize=24, y=0.98)
  fig.subplots_adjust(top=0.85)
  maxyValue = max(maxy)
  #for i in range(n_isto):
  ax.set_ylim((0,maxyValue+100))

def plot_df_means(X1_mean: pd.DataFrame, X2_mean: pd.DataFrame, label1: str, label2: str, title: str, figsize: Tuple[int, int]=(10,5)):
  plt.figure(figsize=figsize)
  plt.bar(X1_mean.index, X1_mean, color='red', label=label1)
  plt.bar(X2_mean.index, X2_mean, color='blue', label=label2)
  plt.xticks(rotation='vertical')
  plt.title(title, fontsize=18)
  plt.legend()
  plt.show()

def plot_explained_variance(explained_variance: List[float], title: str):
  plt.style.use('ggplot')
  plt.figure(figsize=(7,7))
  plt.plot(np.cumsum(explained_variance))
  plt.title(title)
  plt.xlabel('Number of components')
  plt.ylabel('Explained Variance')
  # plot vertical line for  y=variance=90%
  line = plt.gca().get_lines()[0]
  x = line.get_xdata()
  y = line.get_ydata().round(2)
  index_90th = np.where(y >= 0.91)
  x_90th = x[index_90th[0][0]]
  plt.axvline(x=x_90th, marker='.', ls='--', color='blue', label='90% variance or grater')
  plt.legend()
  plt.show()
  return x_90th

def make_meshgrid(X0: pd.DataFrame, X1: pd.DataFrame, h: int):
    """
    Create a mesh of points to be plotted
    :param X0: data to base x-axis meshgrid on
    :param X1:  data to base y-axis meshgrid on
    :param h: stepsize for meshgrid
    :return: nd arrays
    """
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_decision_boundaries(ax: pd.DataFrame, clf: Any, xx: pd.DataFrame, yy: pd.DataFrame, n_features: int, **params):
    """
    Plot decision boundaries of the given classifier
    :param ax: axes of the matplotlib object
    :param clf: classifier
    :param xx: nd array
    :param yy: nd array
    :param params: parameters to pass to countours
    :return: contours
    """
    Xpred = np.array([xx.ravel(), yy.ravel()] + [np.repeat(0, xx.ravel().size) for _ in range(n_features-2)]).T
    Z = clf.predict(Xpred)
    Z = Z.reshape(xx.shape)
    contours = ax.contourf(xx, yy, Z, **params)
    return contours

def decision_boundaries(X_test: pd.DataFrame, y_test: pd.DataFrame, str: str, clf: Any):
  figure, axes = plt.subplots(1, 1)
  print(X_test)
  print("n_features: ",len(X_test.columns))
  X0, X1 = X_test['PC0'], X_test['PC1']
  xx, yy = make_meshgrid(X0, X1, 0.2)
  plot_decision_boundaries(axes, clf, xx, yy, n_features=len(X_test.columns), cmap=plt.cm.coolwarm, alpha=0.8)
  axes.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
  axes.set_xlim(xx.min(), xx.max())
  axes.set_ylim(yy.min(), yy.max())
  axes.set_xlabel('PC1')
  axes.set_ylabel('PC2')
  axes.set_title(str)
  
