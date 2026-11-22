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
# | Created : 22/11/2026 16:51:04      |
# | Updated : 22/11/2026 16:51:04      |
# | END HEADER                         |
# +------------------------------------+

from .layer import Layer
import numpy as np
from .activations import activation_derivative
from .losses import compute_accuracy, precision_recall_f1, get_loss

class Network:
	def __init__(self, tailles, activations):
		self.layers = []
		self.mean = None
		self.std = None
		self.loss = "categoricalCrossentropy"
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

	def backward(self, y_pred, y_true):
		# softmax + cross-entropy (categorique ou binaire, identiques a 2 sorties) : delta = y_pred - y_true
		delta = y_pred - y_true
		for i in range(len(self.layers) - 1, -1, -1):
			delta = self.layers[i].backward(delta)

			if i > 0:
				delta *= activation_derivative(self.layers[i - 1].z, self.layers[i - 1].activation)

	def fit(self, X, y, X_val, y_val, epochs, batch_size, optimizer, patience=0, loss=None):
		history = {
			"loss": [],
			"val_loss": [],
			"acc": [],
			"val_acc": [],
			"precision": [],
			"val_precision": [],
			"recall": [],
			"val_recall": [],
			"f1": [],
			"val_f1": []
		}

		if loss is not None:
			self.loss = loss
		loss_fn = get_loss(self.loss)
		n_samples = X.shape[0]
		best_val_loss = float("inf")
		best_weights = None
		best_epoch = 0
		wait = 0
		for epoch in range(1, epochs + 1):
			indices = np.random.permutation(n_samples)
			X_shuffled = X[indices]
			y_shuffled = y[indices]

			# entrainement par batch_size 
			for i in range(0, n_samples, batch_size):
				X_batch = X_shuffled[i : i + batch_size]
				y_batch = y_shuffled[i : i + batch_size]

				y_pred = self.forward(X_batch)
				self.backward(y_pred, y_batch)
				optimizer.step()

			# eval sur toute les donne a la fin de l epoch
			train_preds = self.predict(X)
			val_preds = self.predict(X_val)
			train_loss = loss_fn(train_preds, y)
			val_loss = loss_fn(val_preds, y_val)
			if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
				raise ValueError("Loss diverged (NaN/inf): lower the learning rate")
			train_acc = compute_accuracy(train_preds, y)
			val_acc = compute_accuracy(val_preds, y_val)
			train_p, train_r, train_f1 = precision_recall_f1(train_preds, y)
			val_p, val_r, val_f1 = precision_recall_f1(val_preds, y_val)
			# sauvegarde
			history["loss"].append(train_loss)
			history["val_loss"].append(val_loss)
			history["acc"].append(train_acc)
			history["val_acc"].append(val_acc)
			history["precision"].append(train_p)
			history["val_precision"].append(val_p)
			history["recall"].append(train_r)
			history["val_recall"].append(val_r)
			history["f1"].append(train_f1)
			history["val_f1"].append(val_f1)

			print(f"epoch {epoch:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
				  f" - val_precision: {val_p:.4f} - val_recall: {val_r:.4f} - val_f1: {val_f1:.4f}")
			if patience > 0:
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					best_epoch = epoch
					best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
					wait = 0
				else:
					wait += 1
					if wait >= patience:
						print(f"early stopping at epoch {epoch} (best epoch: {best_epoch}, val_loss: {best_val_loss:.4f})")
						break
		if best_weights is not None:
			for layer, (W, b) in zip(self.layers, best_weights):
				layer.W = W
				layer.b = b
			print(f"> best weights restored (epoch {best_epoch})")
		history["best_epoch"] = best_epoch if best_weights is not None else None
		return history

	def set_normalization(self, mean, std):
		self.mean = mean
		self.std = std

	def normalize(self, X):
		if self.mean is None or self.std is None:
			raise ValueError("Model has no normalization parameters (retrain it)")
		return (X - self.mean) / self.std

	def predict(self, X):
		return self.forward(X)

	def save(self, chemin):
		model_data = {
			"tailles": [layer.n_in for layer in self.layers] + [self.layers[-1].n_out],
			"activations": [layer.activation for layer in self.layers],
			"weights": [layer.W for layer in self.layers],
			"biases": [layer.b for layer in self.layers],
			"mean": self.mean,
			"std": self.std,
			"loss": self.loss
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
		network.mean = model_data.get("mean")
		network.std = model_data.get("std")
		network.loss = model_data.get("loss", network.loss)
		return network
	