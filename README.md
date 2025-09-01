_For kindred APL projects, see_ [APLAD](https://github.com/BobMcDear/aplearn) _and_ [trap](https://github.com/BobMcDear/trap).

# APLearn

• **[Introduction](#introduction)**<br>
• **[Usage](#usage)**<br>
• **[Available Methods](#available-methods)**<br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Preprocessing Methods](#preprocessing-methods)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Supervised Methods](#supervised-methods)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Unsupervised Methods](#unsupervised-methods)</strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;• <strong>[Miscellaneous](#miscellaneous)</strong><br>
• **[Example](#example)**<br>
• **[Tests](#tests)**<br>

## Introduction

APLearn is a machine learning (ML) library for [Dyalog APL](https://aplwiki.com/wiki/Dyalog_APL) implementing common models as well as utilities for preprocessing data. Inspired by scikit-learn, it offers a bare and intuitive interface that suits the style of the language. Each model adheres to a unified design with two main functionalities, training and prediction/transformation, for seamlessly switching between or composing different methods. One of the chief goals of APLearn is accessibility, particularly for users wishing to modify or explore ML methods in depth without worrying about non-algorithmic, software-focused details.

As argued in the introduction to [trap](https://github.com/BobMcDear/trap) - a similar project implementing the transformer architecture in APL - array programming is an excellent fit for ML and the age of big data. To reiterate, its benefits apropos of these fields include native support for multi-dimensional structures, its data-parallel nature, and an extremely terse syntax that means the mathematics behind an algorithm are directly mirrored in the corresponding code. Of particular importance is the last point since working with ML models in other languages entails either I) Leveraging high-level libraries that conceal the central logic of a program behind walls of abstraction or II) Writing low-level code that pollutes the core definition of an algorithm. This makes it challenging to develop models that can't be easily implemented via the methods supplied by scientific computing packages without sacrificing efficiency. Moreover, tweaking the functionality of existing models becomes impossible in the absence of a comprehensive familiarity with these libraries' enormous and labyrinthine codebases.

For example, scikit-learn is built atop Cython, NumPy, and SciPy, which are themselves written in C, C++, and Fortran. Diving into the code behind a scikit-learn model thus necessitates navigating multiple layers of software, and the low-level pieces are often understandable only to experts. APL, on the other hand, can overcome both these obstacles: Thanks to compilers like [Co-dfns](https://github.com/Co-dfns/Co-dfns) or [APL-TAIL](https://github.com/melsman/apltail), which exploit the data-parallel essence of the language, it can achieve cutting-edge performance, and its conciseness ensures the implementation is to the point and transparent. Therefore, in addition to being a practical instrument that can be used to tackle ML problems, APL/APLearn can be used as tools for better grasping the fundamental principles behind ML methods in a didactic fashion or investigating novel ML techniques more productively.

## Usage

APLearn is organized into four folders: I) Preprocessing methods (```PREPROC```), II) Supervised methods (```SUP```), III) Unsupervised methods (```UNSUP```), and IV) Miscellaneous utilities (```MISC```). In turn, each of these four comprises several components that are discussed further in [the Available Methods section](#available-methods). Most preprocessing, supervised, and unsupervised methods, which are implemented as namespaces, expose two dyadic functions:

* ```fit```: Fits the model and returns its state, which is used during inference. In the case of supervised models, the left argument is the two arrays ```X y```, where ```X``` denotes the independent variables and ```y``` the dependent ones, whereas the only left argument of unsupervised or preprocessing methods is ```X```. The right argument is the hyperparameters.
* ```pred```/```trans```: Predicts or transforms the input data, provided as the left argument, given the model's state, provided as the right argument.

Specifically, each method can be used as seen below for an arbitrary method ```METHOD``` and hyperparameters ```hyps```. There are two exceptions to this rule: ```UNSUP.KMEANS```, an unsupervised method, implements ```pred``` instead of ```trans```, and ```SUP.LDA```, a supervised method, implements ```trans``` in addition to the usual ```pred```.

```apl
⍝ Unupervised/preprocessing; COMP stands for either PREPROC or UNSUP.
st←X y COMP.METHOD.fit hyps
out←X COMP.METHOD.trans st

⍝ Supervised
st←X y SUP.METHOD.fit hyps
out←X SUP.METHOD.pred st
```

## Available Methods

This section lists the methods available in APLearn.

### Preprocessing

* ```PREPROC.NORM```: Normalizes features by subtracting by the mean and dividing by the standard deviation.
  * Hyperparameters: None.
* ```PREPROC.ORD```: Encodes categorical features as integers.
  * Hyperparameters: I) Indices of categorical columns.
* ```PREPROC.ONE_HOT```: Encodes categorical features as one-hot vectors.
  * Hyperparameters: I) Indices of categorical columns.

### Supervised

* ```SUP.RIDGE```: Ridge regression using the closed-form solution.
  * Hyperparameters: I) Positive regularization term
* ```SUP.LASSO```: Lasso regression using coordinate descent.
  * Hyperparameters: I) Regularization term
* ```SUP.LOG_REG```: Logistic regression using L-BFGS.
  * Hyperparameters: I) Regularization term
* ```SUP.LIN_SVC```: Binary linear SVM classifier using SGD.
  * Hyperparameters: I) Learning rate, II) Regularization term
* ```SUP.LDA```: Binary linear discriminant analysis classifier using SVD. In addition to ```init```, ```fit```, and ```pred```, this model also has a ```trans``` function for dimensionality reduction, similar to PCA.
  * Hyperparameters: None
* ```SUP.NB```: Naive Bayes classifier using Gaussian likelihood.
  * Hyperparameters: None
* ```SUP.KNN```: _k_-NN using brute-force search. This model can be used for both classification and regression, indicated by the first hyperparameter passed to the initialization function.
  * Hyperparameters: I) Classification or regression (1 for classification, 0 for regression), II) Number of neighbours
* ```SUP.RF```: Random forest. The underlying tree format of the random forest is the naive, nested one explained [here](https://dfns.dyalog.com/n_Trees.htm) and is inefficient. Improvements will be made in the future. This model can be used for both classification and regression, indicated by the first hyperparameter passed to the initialization function.
  * Hyperparameters: I) Classification or regression (1 for classification, 0 for regression), II) Number of trees, III) Minimum number of samples per split, IV) Number of features to consider at each split

Save for ```SUP.LIN_SVC``` and ```SUP.LDA```, which return actual binary classifications, other classifiers return probability distributions. Regression models output continuous scalars as expected.

### Unsupervised

* ```UNSUP.PCA```: PCA using the SVD-based closed-form solution.
  * Hyperparameters: I) Number of components.
* ```UNSUP.KMEANS```: _k_-means clustering. Unlike ```UNSUP.PCA``` and preprocessing methods,  this model implements ```pred``` in place of ```trans```.
  * Hyperparameters: I) Number of clusters.

### Miscellaneous

Miscellaneous utilities include I) Splitting utilities and II) Metrics. The splitting utilities are:

* ```MISC.SPLIT.train_val```: Creates a train-validation split of the left argument, with the right argument representing the proportion of samples that should constitute the validation set.
* ```MISC.SPLIT.xy```: Separates the independent and dependent variables of the left argument, with the right argument representing the index of the dependent column.


The metrics, whose left and right arguments are respectively the target and predicted values, are:

* ```MISC.METRICS.mae```: Mean absolute error.
* ```MISC.METRICS.mse```: Mean squared error.
* ```MISC.METRICS.rmse```: Root mean squared error.
* ```MISC.METRICS.acc```: Accuracy.
* ```MISC.METRICS.prec```: Precision.
* ```MISC.METRICS.rec```: Recall.
* ```MISC.METRICS.f1```: F1 score.


## Example

The example below showcases a short script employing APLearn to conduct binary classification on [the Adult dataset](https://www.cs.toronto.edu/~delve/data/adult/adultDetail.html). This code is relatively verbose for the sake of explicitness; some of these operations can be composed together for brevity. For instance, the model state could be fed directly to the prediction function, that is, ```out←0⌷⍉⍒⍤1⊢X_v SUP.LOG_REG.pred X_t y_t SUP.LOG_REG.fit 0.01``` instead of two individual lines for training and prediction.

```apl
]Import # APLSource

⍝ Reads data and moves target to first column for ease
(data header)←⎕CSV 'adult.csv' ⍬ 4 1
data header←(header⍳⊂'income')⌽¨data header

⍝ Encodes categorical features and target; target is now last
cat_names←'workclass' 'education' 'marital-status' 'occupation' 'relationship' 'race' 'gender' 'native-country'
data←data PREPROC.ONE_HOT.trans data PREPROC.ONE_HOT.fit header⍳cat_names
data←data PREPROC.ORD.trans data PREPROC.ORD.fit 0

⍝ Creates 80:20 training-validation split and separates input & target
train val←data MISC.SPLIT.train_val 0.2
(X_t y_t) (X_v y_v)←(¯1+≢⍉data) MISC.SPLIT.xy⍨¨train val

⍝ Normalizes data, trains, takes argmax of probabilities, and evaluates accuracy
X_t X_v←(X_t PREPROC.NORM.fit ⍬)∘(PREPROC.NORM.trans⍨)¨X_t X_v
st←X_t y_t SUP.LOG_REG.fit 0.01
out←0⌷⍉⍒⍤1⊢X_v SUP.LOG_REG.pred st
⎕←y_v MISC.METRICS.acc out
```
The final accuracy should be around 85%, which matches the score of the scikit-learn baseline. For more details, please refer to the accompanying [notebook tutorial](https://github.com/BobMcDear/aplearn/blob/main/examples/adults/apl.ipynb). An additional example dealing with regression can be found [here](https://github.com/BobMcDear/aplearn/tree/main/examples/housing), and one treating clustering [here](https://github.com/BobMcDear/aplearn/tree/main/examples/iris).

## Tests

Writing unit tests for this project isn't straightforward since there are no existing baselines. For now, [`tests`](https://github.com/BobMcDear/aplearn/tree/main/tests) merely trains and evaluates APLearn models on synthetic data so that the final scores may be compared to the Python reference. Please see [here](https://github.com/BobMcDear/aplearn/blob/main/tests/README.md) for more information.
