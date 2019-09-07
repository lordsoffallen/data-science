import numpy as np
import tensorflow as tf


class AutoRec(object):

    def __init__(self, visible_dimensions, epochs=200, hidden_dimensions=50, 
                 learning_rate=0.1, batch_size=100):
        """ Create a Autoencoder class instance to train AutoRec network and get recommendations

        Parameters
        ----------
        visible_dimensions: int
            Number of products or movies in this case
        epochs: int
            Number of iterations
        hidden_dimensions: int
            Number of hidden dimensions which defines to latent dimension
        learning_rate: float
            Learning rate of the architecture
        batch_size: int
            Batch size.
        """

        self.visible_dimensions = visible_dimensions
        self.epochs = epochs
        self.hidden_dimensions = hidden_dimensions
        self.learning_rate = learning_rate
        self.batch_size = batch_size    
        
    def train(self, X):
        """ Train the Autoencoder network model.

        Parameters
        ----------
        X: np.ndarray
            Input parameter
        """

        tf.reset_default_graph()
        self.make_graph()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        npX = np.array(X)

        for epoch in range(self.epochs):
            for i in range(0, npX.shape[0], self.batch_size):
                self.sess.run(self.update, feed_dict={self.input_layer: npX[i:i+self.batch_size]})

            print("Trained epoch ", epoch)

    def get_recommendations(self, input_user):
        """ Get recommendations for a given user. Retrives the whole recommendation
        for a user.

        Parameters
        ----------
        input_user: list
            Input user data in a numpy format

        Returns
        -------
        rec: np.ndarray
            Return all missing values on a given user
        """
                 
        # Feed through a single user and return predictions from the output layer.
        rec = self.sess.run(self.output_layer, feed_dict={self.input_layer: input_user})

        return rec[0]         

    def make_graph(self):
        """ Creates a TensorFlow graph for AutoRec. """

        tf.set_random_seed(0)
        
        # Create varaibles for weights for the encoding (visible->hidden) and decoding (hidden->output) stages,
        # randomly initialized
        self.encoder_weights = {'weights': tf.Variable(tf.random_normal([self.visible_dimensions, self.hidden_dimensions]))}
        self.decoder_weights = {'weights': tf.Variable(tf.random_normal([self.hidden_dimensions, self.visible_dimensions]))}
        
        # Create biases
        self.encoder_biases = {'biases': tf.Variable(tf.random_normal([self.hidden_dimensions]))}
        self.decoder_biases = {'biases': tf.Variable(tf.random_normal([self.visible_dimensions]))}
        
        # Create the input layer
        self.input_layer = tf.placeholder('float', [None, self.visible_dimensions])
        
        # hidden layer
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.input_layer, self.encoder_weights['weights']),
                                      self.encoder_biases['biases']))
        
        # output layer for our predictions.
        self.output_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.decoder_weights['weights']),
                                                 self.decoder_biases['biases']))
       
        # Our "true" labels for training are copied from the input layer.
        self.labels = self.input_layer
        
        # loss function and optimizer.
        loss = tf.losses.mean_squared_error(self.labels, self.output_layer)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        
        # What we evaluate each batch.
        self.update = [optimizer, loss]
