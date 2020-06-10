# LimeOut
The project aims to tackle process fairness for Machine Learning Models, while improving (or at-least maintaining the accuracy)
**LimeOut** has two major components :LIME<sub>Global</sub> and the ensemble.
Note that here explanations are in the form of top important features used by the model, while making predictions. (We focused
only on top-10 features, however this hyperparameter may vary as per the use case.)

## LIME<sub>Global</sub>
We propose LIME to obtain global explanations for the ML classifiers. LIME<sub>Global</sub> uses 'submodular pick' approach to 
select the instances, which provide diverse and non-redundant set of LIME explanations. We used these instances to derive the 
global explanations. We used the LIME implementation (as implemented [here](https://github.com/marcotcr/lime/)) with minor 
modifications. 
After obtaining the LIME<sub>Global</sub> explanations, if the sensitive features are among the most imporatant, then the 
classifier is considered as unfair, and we move on to second component of LimeOut. Otherwise, the classifier is deemed to be 
fair and no action is taken,

## Ensemble
This is the core component of the LimeOut. Given the most important features, LimeOut produces a pool of classifiers using 
feature-drop (removing feature from training set). Each of these classifiers does not depend on the corresponding sensitive 
features. It then constructs an ensemble using this pool of classifiers. Following a human and context-centered approach, the 
choice of sensitive features is left to the user within the given context. 

# Experiments
We implemented this approach on Adult([Random Forest Classifier](https://github.com/vaishnavi026/LimeOut/blob/master/Adult_Random_Forest.ipynb)
and [Logistic Regression](https://github.com/vaishnavi026/LimeOut/blob/master/Adult_Logistic_Regression.ipynb)) and 
German Dataset([Random Forest Classifier](https://github.com/vaishnavi026/LimeOut/blob/master/German_Random_Forest.ipynb) and 
[Logistic Regression](https://github.com/vaishnavi026/LimeOut/blob/master/German_Logistic_Regression.ipynb)). These results are
further discussed in the paper.

## Dependencies
* Python >= 3.7
* Scikit-learn >= 0.20.3 (the oldest version where it works)

## Other requirements 
**Package Version**
* anaconda-client                    1.7.2
* anaconda-navigator                 1.9.7
* anaconda-project                   0.8.3
* conda                              4.7.11
* conda-build                        3.18.8
* conda-package-handling             1.3.11
* conda-verify                       3.4.2
* json-tricks                        3.13.5
* json5                              0.8.4
* js onschema                         3.0.1
* jupyter                            1.0.0
* jupyter-client                     5.3.1
* jupyter-console                    6.0.0
* jupyter-core                       4.5.0
* numpy                              1.16.4
* pandas                             0.24.2
* pip                                19.1.1
* scikit-image                       0.15.0
* scikit-learn                       0.21.2
* scipy                              1.3.0
* seaborn                            0.9.0

