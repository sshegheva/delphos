import logging
from sklearn import neighbors
from sklearn import cross_validation

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

n_neighbors = 5


data = None
FEATURES, TARGET = data
# training data
T = None


def fit_classifier():
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(FEATURES, TARGET)
    return clf


def predict(clf, query):
    return clf.predict(query)


def cross_validate_error(df, clf):
    """
    Evaluate classification error
    :param df: data for classification
    :param clf: classifier
    :return: mean score and std deviation of the score
    """
    scores = cross_validation.cross_val_score(clf, df[FEATURES], df[TARGET], cv=5)
    mean_score = scores.mean()
    std_score = scores.std()
    LOGGER.info('Accuracy:  %0.2f (+/- %0.2f)', mean_score, std_score * 2)
    return mean_score, std_score