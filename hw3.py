import math

import numpy
import numpy as np
from scipy.special import factorial


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    if isinstance(k, (list, np.ndarray)):
        # Apply the formula to each element in k, relevant for the test
        k_list = [ki * math.log(rate) - rate - math.log(math.factorial(ki)) for ki in k]
        return np.array(k_list)
    else:
        # Single instance
        return k * math.log(rate) - rate - math.log(math.factorial(k))


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    n = samples.shape[0]
    sum = 0
    for i in range (n):
        sum+= samples[i]

    mean = (1/n)*sum
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    z = norm.ppf(1-alpha/2)
    lower_bound = lambda_mle - z * math.sqrt((lambda_mle/n))
    upper_bound = lambda_mle + z * math.sqrt((lambda_mle/n))

    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    ###NEED TO BE VECTORIZED###
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = []
    sum_likelihood = 0
    for i in rates:
        for sample in samples:
            sum_likelihood += poisson_log_pmf(sample, i)
        likelihoods.append(sum_likelihood)
        sum_likelihood = 0
    return np.array(likelihoods)

def poisson_log_pmf_vectorized(samples, rate):
    ##MIGHT HELP VECTORIZING###
    # samples: shape (n,)
    # rate: scalar
    return samples * np.log(rate) - rate - math.factorial(samples + 1)

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        # I chose it with a,b,c,d parameters that could be anything that maintain the distribution of X,Y,Z separately
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.05,
            (0, 1): 0.25,
            (1, 0): 0.45,
            (1, 1): 0.25
        }  # P(X=x, C=c) =P(C=c)⋅P(X=x∣C=c)

        self.Y_C = {
            (0, 0): 0.25,
            (1, 0): 0.25,
            (0, 1): 0.05,
            (1, 1): 0.45
        }  # P(Y=y, C=c) = P(C=c)⋅P(Y=y∣C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.025,
            (0, 1, 0): 0.025,
            (1, 0, 0): 0.225,
            (1, 1, 0): 0.225,
            (0, 0, 1): 0.025,
            (0, 1, 1): 0.225,
            (1, 0, 1): 0.025,
            (1, 1, 1): 0.225,
        }  # P(X=x, Y=y, C=c) = P(C)⋅P(X∣C)⋅P(Y∣C)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for (x, y), joint_prob in X_Y.items():# Runs on all tuples of 0,1 in X_Y and their possibility (joint_prob)
            marginal_product = self.X[x] * self.Y[y] # makes a vector of p(x)*p(y) for each option
            if not joint_prob == marginal_product:  # small tolerance for float error
                return True  # Found a mismatch → dependent
        return False


    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C

        for x in X:
            for y in Y:
                for c in C:
                    p_xyc = X_Y_C.get((x, y, c), 0)
                    p_xc = X_C.get((x, c), 0)
                    p_yc = Y_C.get((y, c), 0)
                    p_c = C[c]

                    lhs = p_xyc * p_c
                    rhs = p_xc * p_yc

                    if lhs != rhs:
                        return False  # Dependency detected
        return True



def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    instance = 1.0 / (std * math.sqrt(2 * math.pi))
    exponent = -((x - mean) ** 2) / (2 * std ** 2)
    return instance * math.exp(exponent)

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        labels = dataset[:, -1]
        rows = []
        for i in range(len(labels)):
            if labels[i] == class_value:
                rows.append(dataset[i, :-1])

        self.data = np.array(rows)
        self.n_rows = dataset.shape[0]
        self.n_class = self.data.shape[0]

        self.means = np.mean(self.data, axis=0)
        self.stds = np.std(self.data, axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        return self.n_class/self.n_rows
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = 1

        for i in range(x.shape[0]):
            if self.stds[i] < 1e-6: #making sure im not dividing by 0
                self.stds[i] = 1e-6

            likelihood *= normal_pdf(x[i], self.means[i], self.stds[i])

        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """

        return self.get_prior() * self.get_instance_likelihood(x)

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        self.class_dist_0 = ccd0
        self.class_dist_1 = ccd1


    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.class_dist_0.get_instance_joint_prob(x) >= self.class_dist_1.get_instance_joint_prob(x):
            pred = 0
        else:
            pred = 1

        return pred


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    d = mean.shape[0]
    delta = x - mean

    # Regularize covariance for numerical stability (optional but wise)
    cov += 1e-6 * np.eye(d)

    # Compute determinant and inverse
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    # Mahalanobis distance squared
    exponent = -0.5 * (delta.T @ inv_cov @ delta).item()

    # Normalization constant
    norm_const = 1.0 / (math.pow(2 * math.pi, d / 2) * math.sqrt(det_cov))

    # Final PDF
    return norm_const * math.exp(exponent)


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """

        labels = dataset[:, -1]
        rows = []
        for i in range(len(labels)):
            if labels[i] == class_value:
                rows.append(dataset[i, :-1])

        self.data = np.array(rows)
        self.n_rows = dataset.shape[0]
        self.n_class = self.data.shape[0]

        self.means = np.mean(self.data, axis=0)
        self.cov_matrix = np.cov(self.data.T)

        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = self.n_class/self.n_rows
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = multi_normal_pdf(x, self.means, self.cov_matrix)

        return likelihood


    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    count = 0
    n_rows = test_set.shape[0]
    for row in range(n_rows):
        x = test_set[row, :-1]  # feature vector
        true_label = test_set[row, -1]  # actual label
        pred_label = map_classifier.predict(x)

        if pred_label == true_label:
            count += 1

    acc = count/n_rows
    return acc


def class_distribution_maker():
    """
    A function that gets a data set and returns a dictionary with class and
    """


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        labels = dataset[:, -1]
        rows = []
        for i in range(len(labels)):
            if labels[i] == class_value:
                rows.append(dataset[i, :-1])

        self.data = np.array(rows)

        self.n_rows = dataset.shape[0]
        self.n_class = self.data.shape[0]
        self.p = dataset[:, :-1] # number of features = number of columns -1


    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        return self.n_class/self.n_rows
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = 1.0
        num_features = x.shape[0]

        for t in range(num_features):
            v = x[t]

            # Number of times feature t == v in class j
            count_t_v = np.sum(self.data[:, t] == v)

            # Number of possible values for feature t (from full dataset, not just class j)
            V_t = np.unique(self.p[:, t])  # <-- self.p = dataset[:, :-1]
            num_vals = len(V_t)

            # Laplace-smoothed probability
            prob = (count_t_v + 1) / (self.n_class + num_vals)
            likelihood *= prob

        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)

        return joint_prob
