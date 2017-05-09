
==
ml
==

Machine Learning scratches and prototypes. The goal and purpose of the project
is to understand fundamental learning algorithms and try to reproduce them
from scratch and compare with well-known 3rd-party libraries.

In the implementation I've tried to code as much as I could manually
(including matrix operations or their equivalents), but left heavy-lifting
(matrix inverse, eigenvalues problem) to *numpy* and *scipy*. There is also no
data normalization, which affects results to the point, however it makes things
simpler.

Algorithms reproduced (supervised learning):

* linear regression 
* logistic regression
* newtons method
* Gaussian classifier
* naive Bayes classifier
* linear discriminant analysis (LDA)
* multi-layer perceptron classifier (MLP)

*scikit-learn* is used to test and evaluate the implementation: http://scikit-learn.org/

Though its quite hard to find single source of information to cover all topics,
here are some references:

* Stanford machine learning course and its Youtube recordings:
	http://cs229.stanford.edu 
	https://www.youtube.com/view_play_list?p=A89DCFA6ADACE599
* Machine Learning, Tom E. Mitchel, McGraw-Hill Science/Engineering/Math (03/01/1997)
* Various introductory articles scattered across the Internet, e.g.:
	https://ml.berkeley.edu/blog/2016/11/06/tutorial-1/
