import nn
import numpy as np
import matplotlib as plt


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.
        """
        ...

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        ...

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        while True:
            breakWhileLoopNotCheckFromBegining = True
            for x, y in dataset.iterate_once(batch_size):
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    breakWhileLoopNotCheckFromBegining = False
            if breakWhileLoopNotCheckFromBegining:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.hiddenLayerSize = 512
        self.batchSize = 200
        self.learningRate = -0.05
        self.w1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 1)
        self.b2 = nn.Parameter(1, 1)
        self.parameters = [self.w1, self.b1, self.w2, self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        firstLayer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        return nn.AddBias(nn.Linear(firstLayer, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        finalLoss = 1
        while finalLoss > 0.01:
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(gradients[i], self.learningRate)
            finalLoss = nn.as_scalar(loss)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """

    def __init__(self):
        # Initialize your model parameters here
        self.hiddenLayerSize = 200
        self.batchSize = 100
        self.learningRate = -0.5
        self.w1 = nn.Parameter(784, 196)
        self.b1 = nn.Parameter(1, 196)
        self.w2 = nn.Parameter(196, 10)
        self.b2 = nn.Parameter(1, 10)
        self.parameters = [self.w1, self.b1, self.w2,
                           self.b2]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        firstLayer = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        return nn.AddBias(nn.Linear(firstLayer, self.w2), self.b2)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        accuracy = 0
        while accuracy < 0.97:
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(gradients[i], self.learningRate)
            accuracy = dataset.get_validation_accuracy()


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        """
        Our dataset contains words from five different languages, and the
        combined alphabets of the five languages contain a total of 47 unique
        characters.
        """
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters
        self.hiddenSize = 200
        self.batchSize = 1
        self.learningRate = -0.005

        self.w1 = nn.Parameter(self.num_chars, self.hiddenSize)
        self.b1 = nn.Parameter(self.batchSize, self.hiddenSize)
        self.w2 = nn.Parameter(self.num_chars, self.hiddenSize)
        self.b2 = nn.Parameter(self.batchSize, self.hiddenSize)
        self.w3 = nn.Parameter(self.hiddenSize, len(self.languages))
        self.b3 = nn.Parameter(self.batchSize, len(self.languages))
        self.wCalibration = nn.Parameter(self.hiddenSize, self.hiddenSize)
        self.parameters = [self.w1, self.b1,
                           self.w2, self.b2, self.w3, self.b3, self.wCalibration]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for the
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        z0 = nn.Linear(xs[0], self.w1)
        hPrevious = nn.ReLU(nn.AddBias(z0, self.b1))

        for c in xs[1:]:
            z = nn.Add(nn.Linear(hPrevious, self.wCalibration),
                       nn.Linear(c, self.w2))
            h = nn.ReLU(nn.AddBias(z, self.b2))
            hPrevious = h

        return nn.AddBias(nn.Linear(hPrevious, self.w3), self.b3)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        accuracy = 0
        while accuracy < 0.82:
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                for i in range(len(self.parameters)):
                    self.parameters[i].update(gradients[i], self.learningRate)
            accuracy = dataset.get_validation_accuracy()
