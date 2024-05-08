# Step 1: Import Libraries
import numpy as np

# Step 3: Define the Neural Network Architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the network
        self.hidden_output = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output
    
    def backward(self, X, y, output):
        # Backpropagation
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta)
        self.bias_hidden_output += np.sum(self.output_delta, axis=0)
        self.weights_input_hidden += X.T.dot(self.hidden_delta)
        self.bias_input_hidden += np.sum(self.hidden_delta, axis=0)
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output)
    
    def predict(self, X):
        # Make predictions
        return np.round(self.forward(X))



if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # Step 2: Load and Preprocess Data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 4: Train the Model
    input_size = X_train.shape[1]
    hidden_size = 8
    output_size = len(np.unique(y_train))  # Number of classes

    # Convert class labels to one-hot encoded vectors
    def one_hot_encode(labels, num_classes):
        encoded = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            encoded[i, label] = 1
        return encoded

    # Initialize and train the neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, one_hot_encode(y_train, output_size), epochs=1000)

    # Step 5: Evaluate the Model
    predictions = nn.predict(X_test)

    # Convert one-hot encoded predictions back to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, predicted_labels)
    print("Accuracy:", accuracy)

