from Layers.tree import SoftDecisionTree
from Layers.nn import *
import tensorflow as tf
import numpy as np
import os.path
import sys
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime
import datamanager

#AdaBoost
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

mnist_dir = './data/mnist/'
cifar_dir = './data/cifar/'

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross Validation accuracy")

    plt.legend(loc="best")
    return plt




def tf_decisiontree(current_dataset, data):

    checkpoints_dir = "./checkpoints/"
    checkpoints_dir = checkpoints_dir + 'tree/' + data

    if data == "mnist":
        n_features = 784
        n_classes = 10
        
    elif data == "cifar":
        n_features = 3072
        n_classes = 20


   
    batch_size = 32
    val_batch_size = 256

    tree = SoftDecisionTree(max_depth=6,n_features=n_features,n_classes=n_classes,max_leafs=None)
    tree.build_tree()
    #arguments = [tree,checkpoints_dir]
    #t = Thread(target=saveTree, args=(arguments))
   # t.start()

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(tree.loss)

    # Saving the model
    all_saver = tf.train.Saver()
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    EPOCHS = 10000
    TOTAL_BATCH = 16
    display_step = 100
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCHS):

            avg_cost = 0.
           
            # Loop over all batches
            acc =0.0
            val_acc = 0.0
            val_mse = 0.0
            mse = 0.0
            for i in range(TOTAL_BATCH):
                batch_xs, batch_ys = current_dataset.train.next_batch(batch_size)

                c = tree.boost(X=batch_xs,y=batch_ys,sess=sess, optimizer=optimizer)

                target = np.argmax(batch_ys,axis=1)
                preds = tree.predict(X=batch_xs,y=batch_ys,sess=sess)
                acc += accuracy_score(y_pred=preds,y_true=target)/TOTAL_BATCH
                mse += mean_squared_error(y_true=target,y_pred=preds)/TOTAL_BATCH

                # Compute average loss
                avg_cost+= acc/TOTAL_BATCH

            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                batch_val_xs, batch_val_ys = current_dataset.test.next_batch(val_batch_size)


                val_target = np.argmax(batch_val_ys, axis=1)
                val_preds = tree.predict(X=batch_val_xs,y=batch_val_ys,sess=sess)
                val_acc = accuracy_score(y_pred=val_preds, y_true=val_target)
                val_mse = mean_squared_error(y_true=val_target,y_pred=val_preds)


                
                print("Epoch:", '%04d' % (epoch + 1), "training_error=",
                      "{:.9f}".format(mse),"training_accuracy=","{:.4f}".format(acc),"test_error=",
                      "{:.9f}".format(val_mse),"test_accuracy=","{:.4f}".format(val_acc))

                #saves checkpoint to folder
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
                save_path = all_saver.save(sess, checkpoints_dir + "/trained_model.ckpt",global_step=epoch)
            
    sess.close()


def neuralnet(current_dataset, data):
    learning_rate = 0.001
    batch_size = 256
    n_training_epochs = 100

    if data == "mnist":
        n_features = 784
        n_classes = 10
        dim_1 = 28
        dim_2 = 28
        dim_3 = 1
        

    elif data == "cifar":
        n_features = 3072
        n_classes = 20
        dim_1 = 32 * 3
        dim_2 = 32
        dim_3 = 1
        

    reshape_size = n_features * n_classes

    # Input (X) and Target (Y) placeholders, they will be fed with a batch of
    # input and target values respectively, from the training and test sets
    X = input_placeholder(n_features)
    Y = target_placeholder(n_classes)

    #convolutional neural network
    logits_op, preds_op, loss_op = \
        convnet(tf.reshape(X, [-1, dim_1, dim_2, dim_3]), Y, convlayer_sizes=[n_classes, n_classes],
                     filter_shape=[3, 3], outputsize=n_classes, padding="same", reshape_size=reshape_size)
    tf.summary.histogram('pre_activations', logits_op)

   # The training op performs a step of stochastic gradient descent on a minibatch
    optimizer = tf.train.AdamOptimizer  # ADAM - widely used optimiser (ref: http://arxiv.org/abs/1412.6980)
    train_op = optimizer(learning_rate).minimize(loss_op)

    # Prediction and accuracy ops
    accuracy_op = get_accuracy_op(preds_op, Y)

    # TensorBoard for visualisation
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    summaries_op = tf.summary.merge_all()

    # Separate accuracy summary so we can use train and test sets
    accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
    accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

    # When run, the init_op initialises any TensorFlow variables
    # hint: weights and biases in our case
    init_op = tf.global_variables_initializer()

    # Get started
    sess = tf.Session()
    sess.run(init_op)

    # Initialise TensorBoard Summary writers
    dtstr = "{:%b_%d_%H-%M-%S}".format(datetime.now())
    train_writer = tf.summary.FileWriter('./summaries/' + dtstr + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./summaries/' + dtstr + '/test')

    # Train
    print('Starting Training...')
    train_accuracy, test_accuracy = nn_train(sess, current_dataset, n_training_epochs, batch_size,
                                          summaries_op, accuracy_summary_op, train_writer, test_writer,
                                          X, Y, train_op, loss_op, accuracy_op, accuracy_placeholder)
    print('Training Complete\n')
    print("train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}".format(**locals()))

    # Clean up
    sess.close()

def sk_decisiontree(current_dataset, data):
    from sklearn.tree import DecisionTreeClassifier
    training_batch_size = 15000
    title = "Decision Tree - " + data.upper() + " (Learning Curve)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    clf = DecisionTreeClassifier(random_state=0)
    batch_xs, batch_ys = current_dataset.train.next_batch(training_batch_size)
    plot_learning_curve(clf, title, batch_xs, batch_ys, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

    test_batch_size = training_batch_size // 2
    batch_test_xs, batch_test_ys = current_dataset.test.next_batch(test_batch_size)
    model = clf.fit(batch_xs, batch_ys)
    batch_test_ys_pred = model.predict(batch_test_xs)
    testacc = metrics.accuracy_score(batch_test_ys, batch_test_ys_pred)
    print("Decision Tree -" + data + " Test Accuracy:", testacc)

def adaboost(current_dataset, data):
    from sklearn.ensemble import AdaBoostClassifier
    training_batch_size = 30000
    title = "Adaboost - " + data.upper() + " (Learning Curve)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    #svclassifier = SVC(gamma=0.001)
    batch_xs, batch_ys = current_dataset.train.next_batch(training_batch_size)
    plot_learning_curve(abc, title, batch_xs, batch_ys, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()
    
    test_batch_size = training_batch_size // 2
    batch_test_xs, batch_test_ys = current_dataset.test.next_batch(test_batch_size)
    model = abc.fit(batch_xs, batch_ys)
    batch_test_ys_pred = model.predict(batch_test_xs)
    testacc = metrics.accuracy_score(batch_test_ys, batch_test_ys_pred)
    print("Adaboost -" + data + " Test Accuracy:", testacc)

    
def bagboost(current_dataset, data):
    from sklearn.ensemble import BaggingClassifier
    training_batch_size = 8000

    title = "Bagging - " + data.upper() + " (Learning Curve)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    bag = BaggingClassifier(n_estimators=50)
    #svclassifier = SVC(gamma=0.001)
    batch_xs, batch_ys = current_dataset.train.next_batch(training_batch_size)
    plot_learning_curve(bag, title, batch_xs, batch_ys, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

    test_batch_size = training_batch_size // 2
    batch_test_xs, batch_test_ys = current_dataset.test.next_batch(test_batch_size)
    model = bag.fit(batch_xs, batch_ys)
    batch_test_ys_pred = model.predict(batch_test_xs)
    testacc = metrics.accuracy_score(batch_test_ys, batch_test_ys_pred)
    print("Bagboost -" + data + " Test Accuracy:", testacc)
    
def svmkernel(current_dataset, data):
    from sklearn.svm import SVC
    training_batch_size = 2500
    title = "SVM Kernel Linear - " + data.upper() + " (Learning Curve)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    svclassifier = SVC(kernel='linear')
    #svclassifier = SVC(gamma=0.001)
    batch_xs, batch_ys = current_dataset.train.next_batch(training_batch_size)
    plot_learning_curve(svclassifier, title, batch_xs, batch_ys, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()
    
    test_batch_size = training_batch_size // 2
    batch_test_xs, batch_test_ys = current_dataset.test.next_batch(test_batch_size)
    model = svclassifier.fit(batch_xs, batch_ys)
    batch_test_ys_pred = model.predict(batch_test_xs) 
    testacc = metrics.accuracy_score(batch_test_ys, batch_test_ys_pred)
    print("SVM -" + data + " Test Accuracy:", testacc)


def kneighbor(current_dataset, data):
    from sklearn.neighbors import KNeighborsClassifier

    n_neighbors = 3
    training_batch_size = 1000
    test_batch_size = 1000
    title = "K=" + n_neighbors + " Nearest Neighbors - " + data.upper() + " (Learning Curve)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    batch_xs, batch_ys = current_dataset.train.next_batch(training_batch_size)
    plot_learning_curve(knn, title, batch_xs, batch_ys, (0.0, 0.5), cv=cv, n_jobs=4)
    plt.show()

    test_batch_size = training_batch_size // 2
    batch_test_xs, batch_test_ys = current_dataset.test.next_batch(test_batch_size)
    model = knn.fit(batch_xs, batch_ys)
    batch_test_ys_pred = model.predict(batch_test_xs) 
    testacc = metrics.accuracy_score(batch_test_ys, batch_test_ys_pred)
    print("K=" + n_neighbors + " Nearest Neighbors - " + data + " Test Accuracy:", testacc)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["tree","nn", "ada","bag","svm", "knear"])
    parser.add_argument("data", choices=["mnist", "cifar"])

    args = parser.parse_args()

    group1 = ["tree", "nn"]

    if args.method in group1:
        if args.data == "mnist":
            current_dataset = datamanager.mnist_read_data_sets(train_dir=mnist_dir, one_hot=True, reshape=True)
        elif args.data == "cifar":
            current_dataset = datamanager.cifar_read_data_sets(train_dir=cifar_dir,one_hot=True, reshape=False, n_classes=20)  
    else:
        if args.data == "mnist":
            current_dataset = datamanager.mnist_read_data_sets(train_dir=mnist_dir, one_hot=False, reshape=True)
        elif args.data == "cifar":
            current_dataset = datamanager.cifar_read_data_sets(train_dir=cifar_dir,one_hot=False, reshape=False, n_classes=20)  
                


    #if args.method == "tree":
    #    tf_decisiontree(current_dataset, args.data)
    elif args.method == "tree":
        sk_decisiontree(current_dataset, args.data)
    elif args.method == "nn":
        neuralnet(current_dataset, args.data)
    elif args.method == "ada":
        adaboost(current_dataset, args.data)
    elif args.method == "bag":
        bagboost(current_dataset, args.data)
    elif args.method == "svm":
        svmkernel(current_dataset, args.data)
    elif args.method == "knear":
        kneighbor(current_dataset, args.data)





