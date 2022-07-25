import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if (nn.as_scalar(self.run(x)) < 0):
            return -1
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        error = True
        correct = 0
        incorrect = 0
        while error:
            error = False
            for x, y in dataset.iterate_once(batch_size):
                prediction = self.get_prediction(x)
                if prediction - nn.as_scalar(y) != 0:
                    self.w.update(x, -prediction)
                    incorrect += 1
                    error = True
                else:
                    correct += 1


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #Two layer neural network
        self.m0 = nn.Parameter(1,80)
        self.b0 = nn.Parameter(1,80)
        self.m1 = nn.Parameter(80,1)
        self.b1 = nn.Parameter(1,1)
        self.learning_rate = 0.01

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xm1 = nn.Linear(x, self.m0)
        r1 = nn.ReLU(nn.AddBias(xm1, self.b0))
        xm2 = nn.Linear(r1, self.m1)
        predicted_y = nn.AddBias(xm2,self.b1)
        return predicted_y


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 10
        multiplier = 0.01
        error = True
        while error:
            error = False
            streak = 0
            for x, y in dataset.iterate_once(batch_size):
                loss_node = self.get_loss(x, y)
                if (nn.as_scalar(loss_node) > 0.02):
                    gradient = nn.gradients(loss_node, [self.m0, self.m1, self.b0, self.b1])
                    self.m0.update(gradient[0], -multiplier)
                    self.m1.update(gradient[1], -multiplier)
                    self.b0.update(gradient[2], -multiplier)
                    self.b1.update(gradient[3], -multiplier)
                    error = True
                #added this just to see when it terminates
                elif streak > 50:
                    error = False
                    break
                else:
                    streak += 1



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.m0 = nn.Parameter(784, 100)
        self.b0 = nn.Parameter(1, 100)
        self.m1 = nn.Parameter(100, 10)
        self.b1 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xm1 = nn.Linear(x, self.m0)
        r1 = nn.ReLU(nn.AddBias(xm1, self.b0))
        xm2 = nn.Linear(r1, self.m1)
        predicted_y = nn.AddBias(xm2, self.b1)
        return predicted_y

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
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        multiplier = 0.1
        batch_size = 5
        keep_going = True
        while keep_going:
            for x, y in dataset.iterate_once(batch_size):
                loss_node = self.get_loss(x, y)

                gradient = nn.gradients(loss_node, [self.m0, self.m1, self.b0, self.b1])
                self.m0.update(gradient[0], -multiplier)
                self.m1.update(gradient[1], -multiplier)
                self.b0.update(gradient[2], -multiplier)
                self.b1.update(gradient[3], -multiplier)
            if dataset.get_validation_accuracy() >= 0.973:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layers = 300
        self.w0 = nn.Parameter(self.num_chars, self.hidden_layers)
        self.b0 = nn.Parameter(1, self.hidden_layers)
        self.w1 = nn.Parameter(self.hidden_layers, self.hidden_layers)
        self.b1 = nn.Parameter(1, self.hidden_layers)
        self.w2 = nn.Parameter(self.hidden_layers, len(self.languages))
        self.b2 = nn.Parameter(1, len(self.languages))

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

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
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
        "*** YOUR CODE HERE ***"
        # initial z = xiW
        z = nn.Linear(xs[0], self.w0)
        # add non linearity for f initial
        z = nn.ReLU(nn.AddBias(z, self.b0))


        # for hidden layers of consequent characters
        for x in xs[1:]:
            z = nn.Add(nn.ReLU(nn.AddBias(nn.Linear(z, self.w1), self.b1)),
                       nn.ReLU(nn.AddBias(nn.Linear(x, self.w0), self.b0)))
            z = nn.Add(nn.ReLU(nn.Linear(z, self.w1)),
                       nn.ReLU(nn.Linear(x, self.w0)))
            """
            z1 = nn.Linear(xs[i + 1], self.w0)
            r1 = nn.ReLU(nn.AddBias(z1, self.b0))
            z2 = nn.Linear(r1, self.w1)
            r1 = nn.ReLU(nn.AddBias(z1, self.b1))
            z2 = nn.Linear(r1, self.w2)
            z = nn.Add(z, z2)
            z1 = z2
        """

        predicted_y = nn.AddBias(nn.Linear(z, self.w2), self.b2)
        return predicted_y

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
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(xs), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        multiplier = 0.01
        batch_size = 10
        keep_going = True
        while keep_going:
            for x, y in dataset.iterate_once(batch_size):
                loss_node = self.get_loss(x, y)

                gradient = nn.gradients(loss_node, [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2])
                self.w0.update(gradient[0], -multiplier)
                self.w1.update(gradient[1], -multiplier)
                self.w2.update(gradient[2], -multiplier)
                self.b0.update(gradient[3], -multiplier)
                self.b1.update(gradient[4], -multiplier)
                self.b2.update(gradient[5], -multiplier)

            if dataset.get_validation_accuracy() >= 0.85:
                return
