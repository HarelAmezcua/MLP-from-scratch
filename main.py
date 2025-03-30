import numpy as np

# --------------------------------------------------
# 1. Generación de datos de ejemplo
# --------------------------------------------------
# En este ejemplo crearemos un conjunto de datos sintético
# para un problema de clasificación binaria sencillo.

def generar_datos(n_muestras=200):
    # Generamos n_muestras/2 de un tipo y n_muestras/2 de otro.
    # Aquí la idea es tener dos "nubes" de puntos.
    X = np.zeros((n_muestras, 2))
    y = np.zeros((n_muestras, 1))

    # Primer conjunto de puntos (etiqueta = 0)
    X[:n_muestras//2, 0] = np.random.normal(2, 1, n_muestras//2)
    X[:n_muestras//2, 1] = np.random.normal(2, 1, n_muestras//2)
    y[:n_muestras//2, 0] = 0

    # Segundo conjunto de puntos (etiqueta = 1)
    X[n_muestras//2:, 0] = np.random.normal(-2, 1, n_muestras//2)
    X[n_muestras//2:, 1] = np.random.normal(-2, 1, n_muestras//2)
    y[n_muestras//2:, 0] = 1

    return X, y

# --------------------------------------------------
# 2. Definición de la clase MLP
# --------------------------------------------------
class MLP:
    def __init__(self, sizes, learning_rate=0.01):
        """
        Parámetros:
        -----------
        sizes: lista de tamaños [input_dim, hidden1, hidden2, ..., output_dim].
        learning_rate: tasa de aprendizaje para el descenso de gradiente.
        """
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.num_layers = len(sizes) - 1  # Número de capas de pesos

        # Inicializar pesos y sesgos (bias) aleatoriamente
        self.params = self.inicializar_pesos()

    def inicializar_pesos(self):
        """
        Retorna un diccionario con los pesos y biases:
        {
            'W1': matriz de pesos capa 1,
            'b1': vector de bias capa 1,
            'W2': matriz de pesos capa 2,
            'b2': vector de bias capa 2,
            ...
        }
        """
        np.random.seed(42)  # Semilla para reproducibilidad
        params = {}
        for i in range(self.num_layers):
            limit = np.sqrt(2.0 / self.sizes[i]) 
            params['W' + str(i+1)] = np.random.randn(self.sizes[i], self.sizes[i+1]) * limit
            params['b' + str(i+1)] = np.zeros((1, self.sizes[i+1]))
        return params

    def forward(self, X):
        """
        Realiza el paso hacia adelante (forward pass).
        Retorna las activaciones de cada capa en un diccionario.
        """
        activations = {'A0': X}
        A = X

        for i in range(self.num_layers):
            W = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]
            Z = np.dot(A, W) + b  # pre-activación
            # Puedes cambiar la función de activación según convenga
            A = self.relu(Z) if i < self.num_layers - 1 else self.sigmoid(Z)
            activations['Z' + str(i+1)] = Z
            activations['A' + str(i+1)] = A

        return activations

    def backward(self, activations, y):
        """
        Realiza el paso hacia atrás (backpropagation).
        Retorna un diccionario con los gradientes dW y db para cada capa.
        """
        grads = {}
        m = y.shape[0]  # número de muestras

        # Obtener la última activación
        A_final = activations['A' + str(self.num_layers)]

        # Cálculo de la pérdida (binary cross-entropy para este ejemplo):
        # L = -1/m * sum( y*log(A_final) + (1-y)*log(1-A_final) )
        # En este ejemplo, no retornamos la pérdida como tal,
        # pero si lo quieres, podrías calcularla aquí.

        # Derivada de la pérdida con respecto a la salida
        dA = -( (y / (A_final + 1e-8)) - ((1 - y) / (1 - A_final + 1e-8)) )

        for i in reversed(range(self.num_layers)):
            # Variables auxiliares
            A_prev = activations['A' + str(i)]
            Z = activations['Z' + str(i+1)]
            W = self.params['W' + str(i+1)]

            if i == self.num_layers - 1:
                # Capa de salida: la activación es sigmoid
                dZ = dA * self.sigmoid_deriv(Z)
            else:
                # Capa oculta: la activación utilizada es ReLU
                dZ = dA * self.relu_deriv(Z)

            # Gradientes con respecto a W y b
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            grads['dW' + str(i+1)] = dW
            grads['db' + str(i+1)] = db

            # Propagar el error hacia la capa anterior
            dA = np.dot(dZ, W.T)

        return grads

    def actualizar_pesos(self, grads):
        """
        Actualiza los pesos y sesgos utilizando gradientes y descenso de gradiente.
        """
        for i in range(self.num_layers):
            self.params['W' + str(i+1)] -= self.learning_rate * grads['dW' + str(i+1)]
            self.params['b' + str(i+1)] -= self.learning_rate * grads['db' + str(i+1)]

    def train(self, X, y, epochs=1000):
        """
        Entrena el MLP en el conjunto de datos X, y.
        """
        for epoch in range(epochs):
            # 1) Forward
            activations = self.forward(X)
            # 2) Backward
            grads = self.backward(activations, y)
            # 3) Actualización de pesos
            self.actualizar_pesos(grads)

            if (epoch+1) % 100 == 0:
                # Cálculo de la pérdida para monitorear
                loss = self.calcular_loss(activations['A' + str(self.num_layers)], y)
                print(f"Epoch {epoch+1}/{epochs} - Pérdida: {loss:.4f}")

    def predict(self, X):
        """
        Retorna las predicciones (0 o 1) para las entradas X.
        """
        activations = self.forward(X)
        A_final = activations['A' + str(self.num_layers)]
        return (A_final >= 0.5).astype(int)

    def calcular_loss(self, y_pred, y_true):
        """
        Calcula la pérdida de entropía cruzada binaria (binary cross entropy).
        """
        m = y_true.shape[0]
        # Evitamos log(0) usando un pequeño epsilon
        epsilon = 1e-8
        loss = -1/m * np.sum(y_true * np.log(y_pred + epsilon) +
                             (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    # --------------------------------------------------
    # Funciones de activación
    # --------------------------------------------------
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_deriv(Z):
        # derivada(sigmoid(Z)) = sigmoid(Z)*(1 - sigmoid(Z))
        sig = 1 / (1 + np.exp(-Z))
        return sig * (1 - sig)

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_deriv(Z):
        return (Z > 0).astype(float)

# --------------------------------------------------
# 3. Ejemplo de uso
# --------------------------------------------------
if __name__ == "__main__":
    # Generamos datos
    X, y = generar_datos(200)

    # Definimos la arquitectura de la red:
    # - 2 neuronas de entrada (por tener X con 2 características)
    # - 4 neuronas en la capa oculta (puedes ajustar este número)
    # - 1 neurona de salida (clasificación binaria)
    mlp = MLP(sizes=[2, 10,10,10, 1], learning_rate=0.01)

    # Entrenamiento
    mlp.train(X, y, epochs=10000)

    # Predicciones en el mismo conjunto (overfitting intencional, solo para ejemplo)
    preds = mlp.predict(X)
    accuracy = np.mean(preds == y)
    print(f"Exactitud en el conjunto de entrenamiento: {accuracy * 100:.2f}%")
