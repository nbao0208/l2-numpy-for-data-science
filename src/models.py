import numpy as np

class SoftmaxRegression:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        Z = X @ self.W + self.b
        return self._softmax(Z)
    
    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def compute_loss(self, y_pred, y):
        m = y.shape[0]
        Y_one_hot = np.eye(y_pred.shape[1])[y]
        loss = -np.sum(Y_one_hot * np.log(y_pred + 1e-15)) / m
        return loss

    def compute_gradients(self, X, A, y):
        m = X.shape[0]
        Y_one_hot = np.eye(A.shape[1])[y]
        dZ = (A - Y_one_hot) / m
        dW = X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        return dW, db

    def fit(self, X, y, epochs, lr):        
        for i in range(epochs):
            A = self.forward(X)
            loss = self.compute_loss(A, y)
            dW, db = self.compute_gradients(X, A, y)
            
            # Cập nhật tham số
            self.W -= lr * dW
            self.b -= lr * db

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        A = self.forward(X)
        return np.argmax(A, axis=1)
    
    def save(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, in_features, out_features):
        model = cls(in_features, out_features)
        data = np.load(filepath)
        model.W = data['W']
        model.b = data['b']
        return model
    