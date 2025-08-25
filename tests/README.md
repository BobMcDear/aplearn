# APLearn Tests

Because there is no prior machine learning work in APL, unit testing isn't easy. The closest we can get is to employ an interface like [Py'n'APL](https://github.com/Dyalog/pynapl) to exchange data, models, and outputs between APL and Python, but that requires significant effort given Py'n'APL's current state. Here, as a proxy, we simply train and evaluate APLearn models on synthetically-generated data and compare the final scores to the scikit-learn baseline. Currently, there are two separate collection of tests for classification and regression. The Python evaluation results are as follows:

* Classification (accuracy):
    * Linear SVC: 0.86
    * LDA: 0.80
    * _k_-NN: 0.87
    * Logistic regression: 0.85
    * Naive Bayes: 0.83
    * Random forest: 0.98

* Regression (root mean squared error):
    * _k_-NN: 31
    * Ridge: 10
    * Lasso: 12
    * Random forest: 18

Modest variability in performance is normal since some of these models involve stochastic processes.
