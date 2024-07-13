import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pickle import dump

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    with open("filename.pkl", "wb") as f:
        dump(model, f, protocol=5)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidences = []
    labels = []

    # Dictionary of Months with their expected value in evidences
    months = {
        'Jan': 0,
        'Feb': 1,
        'Mar': 2,
        'Apr': 3,
        'May': 4,
        'June': 5,
        'Jul': 6,
        'Aug': 7,
        'Sep': 8,
        'Oct': 9,
        'Nov': 10,
        'Dec': 11
    }

    # Read the csv file
    with open(file=filename) as csvFile:
        reader = csv.reader(csvFile)

        rowNum = 0

        # Loop through each row
        for row in reader:
            # Skip the first row
            if rowNum == 0:
                rowNum += 1
                continue

            monthIndex = 10
            visitorTypeIndex = 15
            weekendIndex = 16

            label = row.pop(-1)  # Last column

            row[monthIndex] = months[row[monthIndex]]
            row[visitorTypeIndex] = 1 if row[visitorTypeIndex] == "Returning_Visitor" else 0
            row[weekendIndex] = 1 if row[weekendIndex] == "TRUE" else 0

            evidence = []

            # Convert each columns to int or float based on it's specification
            for col in row:
                try:
                    val = int(col)
                except ValueError:
                    val = float(col)

                evidence.append(val)

            labels.append(1 if label == "TRUE" else 0)
            evidences.append(evidence)

    return (evidences, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    classifier = KNeighborsClassifier(n_neighbors=1)

    return classifier.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    labelsNum = len(labels)

    truePositives = 0
    falsePositives = 0
    trueNegativs = 0
    falseNegatives = 0

    for i in range(labelsNum):
        # Make sure that the values are always a integer
        labelVal = int(labels[i])
        predictionVal = int(predictions[i])

        if labelVal == 1 and predictionVal == 1:
            # Model correctly predicted a positive value
            truePositives += 1
        elif labelVal == 1 and predictionVal == 0:
            # Model incorrectly predicted a negative value
            falseNegatives += 1
        elif labelVal == 0 and predictionVal == 1:
            # Model incorrectly predicted a positive value
            falsePositives += 1
        elif labelVal == 0 and predictionVal == 0:
            # Model correctly predicted a negative value
            trueNegativs += 1

    sensitivity = truePositives / (truePositives + falseNegatives)
    specificity = trueNegativs / (trueNegativs + falsePositives)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
