import numpy as np
import tensorflow as tf


class RBM(object):

    def __init__(self, visible_dimensions, epochs=20, hidden_dimensions=50, 
                 rating_values=10, learning_rate=0.001, batch_size=100):
        """ Create a RBM class instance to train RBM network and get recommendations

        Parameters
        ----------
        visible_dimensions: int
            Number of products or movies in this case
        epochs: int
            Number of iterations
        hidden_dimensions: int
            Number of hidden dimensions which defines to latent dimension
        rating_values: int
            Number of distinct rating values
        learning_rate: float
            Learning rate of the architecture
        batch_size: int
            Batch size.
        """

        self.visible_dimensions = visible_dimensions
        self.epochs = epochs
        self.hidden_dimensions = hidden_dimensions
        self.rating_values = rating_values
        self.learning_rate = learning_rate
        self.batch_size = batch_size     
                
    def train(self, X):
        """ Train the RBM network model.

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

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            
            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batch_size):
                self.sess.run(self.update, feed_dict={self.X: trX[i:i+self.batch_size]})

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
                 
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hidden_bias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visible_bias)

        feed = self.sess.run(hidden, feed_dict={self.X: input_user})
        rec = self.sess.run(visible, feed_dict={hidden: feed})
        return rec[0]       

    def make_graph(self):
        """ Creates a TensorFlow graph for RBM. """

        tf.set_random_seed(0)
        
        # Create variables for the graph, weights and biases
        self.X = tf.placeholder(tf.float32, shape=[None, self.visible_dimensions], name="X")
        
        # Initialize weights randomly
        self.weights = tf.Variable(tf.random_uniform([self.visible_dimensions, self.hidden_dimensions]), 
                                   dtype=tf.float32, name="weights")
        
        self.hidden_bias = tf.Variable(tf.zeros([self.hidden_dimensions], dtype=tf.float32, name="hiddenBias"))
        self.visible_bias = tf.Variable(tf.zeros([self.visible_dimensions], dtype=tf.float32, name="visibleBias"))
        
        # Perform Gibbs Sampling for Contrastive Divergence, per the paper we assume k=1 instead of iterating over the 
        # forward pass multiple times since it seems to work just fine

        # Forward pass
        # Sample hidden layer given visible...
        # Get tensor of hidden probabilities
        h_prob0 = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hidden_bias)

        # Sample from all of the distributions
        h_sample = tf.nn.relu(tf.sign(h_prob0 - tf.random_uniform(tf.shape(h_prob0))))
        
        # Stitch it together
        forward = tf.matmul(tf.transpose(self.X), h_sample)
        
        # Backward pass
        # Reconstruct visible layer given hidden layer sample
        v = tf.matmul(h_sample, tf.transpose(self.weights)) + self.visible_bias
        
        # Build up our mask for missing ratings
        v_mask = tf.sign(self.X) # Make sure everything is 0 or 1

        # Reshape into arrays of individual ratings
        v_mask3D = tf.reshape(v_mask, [tf.shape(v)[0], -1, self.rating_values])

        # Use reduce_max to either give us 1 for ratings that exist, and 0 for missing ratings
        v_mask3D = tf.reduce_max(v_mask3D, axis=[2], keepdims=True) 
        
        # Extract rating vectors for each individual set of 10 rating binary values
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.rating_values])

        v_prob = tf.nn.softmax(v * v_mask3D) # Apply softmax activation function

        # And shove them back into the flattened state. Reconstruction is done now.
        v_prob = tf.reshape(v_prob, [tf.shape(v)[0], -1]) 

        # Stitch it together to define the backward pass and updated hidden biases
        h_prob1 = tf.nn.sigmoid(tf.matmul(v_prob, self.weights) + self.hidden_bias)
        backward = tf.matmul(tf.transpose(v_prob), h_prob1)
    
        # Now define what each epoch will do...
        # Run the forward and backward passes, and update the weights
        weight_update = self.weights.assign_add(self.learning_rate * (forward - backward))

        # Update hidden bias, minimizing the divergence in the hidden nodes
        hidden_bias_update = self.hidden_bias.assign_add(self.learning_rate * tf.reduce_mean(h_prob0 - h_prob1, 0))

        # Update the visible bias, minimizng divergence in the visible results
        visible_bias_update = self.visible_bias.assign_add(self.learning_rate * tf.reduce_mean(self.X - v_prob, 0))

        self.update = [weight_update, hidden_bias_update, visible_bias_update]
