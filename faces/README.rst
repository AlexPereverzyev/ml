
Faces
=====

The project is a practical iterative learning example of training SVM
classifier to detect human faces. It consists of a several scripts which
prepare data, train and test model. Finally, there is a script to try the
model on target data set and feed selected samples back to training data set.

All the mentioned is implemented on top of scikit-learn framework.

Iterative Learning
==================

Is a process of iteratively enriching training data with preselected samples
in order to improve model accuracy. In case of frontal face detection, first 
training is performed on a limited data set, then classifier is applied to
target image set to produce false positives. The false positives is then used
to enrich training data set before the next learning iteration. In other words,
the negative feedback trains classifier to 'ignore' background noise. 

1. TRAIN(DATA) = MODEL
2. APPLY(MODEL, TARGET_SET] = [NEGATIVE, POSITIVE]
3. DATA' = DATA + [NEGATIVE, POSITIVE*]
4. TRAIN(DATA') = MODEL'

 ...and so on. Here POSITIVE* is a portion of available faces data set.
 