# +------------------------------------+
# | CHAT-DISPARU // HEADER             |
# |       _---~~(~~-_.                 |
# |     _{        )   )                |
# |   ,   ) -~~- ( ,-' )_              |
# |  (  `-,_..`., )-- '_,)             |
# | ( ` _)  (  -~( -_ `,  }            |
# | (_-  _  ~_-~~~~`,  ,' )            |
# |   `~ -^(    __;-,((()))            |
# |         ~~~~ {_ -_(())             |
# |                `\  }               |
# |                  { }               |
# |                                    |
# | File    : network.py               |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 04/11/2026 12:57:17      |
# | Updated : 04/11/2026 12:57:17      |
# | END HEADER                         |
# +------------------------------------+

from .layer import Layer
import numpy as np
from .losses import binary_cross_entropy, compute_accuracy
from .activations import relu, sigmoid, softmax, activation_derivative


class Network:
	def __init__(self, tailles, activations):
		self.layers = []

		for i in range(len(tailles) - 1):
			layer = Layer(
				tailles[i],
				tailles[i + 1],
				activation=activations[i]
			)
			self.layers.append(layer)

	def forward(self, X):
		a = X
		for layer in self.layers:
			a = layer.forward(a)
		return a
	def backward(self, y_pred, y_true, lr):
		delta = y_pred - y_true
		for i in range(len(self.layers) - 1, -1, -1):
			delta = self.layers[i].backward(delta, lr)

			if i > 0:
				delta *= activation_derivative(self.layers[i - 1].z, self.layers[i - 1].activation)
	def fit(self, X, y, X_val, y_val, epochs, batch_size, lr):
		#dico enregistre les metrique a chaqew epoch
		history = {
			"loss": [],
			"val_loss": [],
			"acc": [],
			"val_acc": []
		}

		n_samples = X.shape[0]

		for epoch in range(1, epochs + 1):
			indices = np.random.permutation(n_samples)
			X_shuffled = X[indices]
			y_shuffled = y[indices]

			# entrainement par batch_size 
			for i in range(0, n_samples, batch_size):
				X_batch = X_shuffled[i : i + batch_size]
				y_batch = y_shuffled[i : i + batch_size]

				y_pred = self.forward(X_batch)
				self.backward(y_pred, y_batch, lr)

			#eval sur toute les donne a la fin de l epoch
			train_preds = self.predict(X)
			val_preds = self.predict(X_val)
			train_loss = binary_cross_entropy(train_preds, y)
			val_loss = binary_cross_entropy(val_preds, y_val)
			train_acc = compute_accuracy(train_preds, y)
			val_acc = compute_accuracy(val_preds, y_val)
			#sauvegarde
			history["loss"].append(train_loss)
			history["val_loss"].append(val_loss)
			history["acc"].append(train_acc)
			history["val_acc"].append(val_acc)
			#truc du sujet
			print(f"epoch {epoch:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
		return history
	def predict(self, X):
		return self.forward(X)
	def save(self, chemin):
		model_data = {
			"tailles": [layer.n_in for layer in self.layers] + [self.layers[-1].n_out],
			"activations": [layer.activation for layer in self.layers],
			"weights": [layer.W for layer in self.layers],
			"biases": [layer.b for layer in self.layers]
		}
		np.save(chemin, model_data, allow_pickle=True)
		print(f"> saving model '{chemin}' to disk...")

	@classmethod
	def load(cls, chemin):
		model_data = np.load(chemin, allow_pickle=True).item()
		network = cls(model_data["tailles"], model_data["activations"])
		for layer, W, b in zip(network.layers, model_data["weights"], model_data["biases"]):
			layer.W = W
			layer.b = b
		return network