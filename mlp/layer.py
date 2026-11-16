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
# | File    : layer.py                 |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 04/11/2026 12:57:11      |
# | Updated : 04/11/2026 12:57:11      |
# | END HEADER                         |
# +------------------------------------+


import numpy as np
from .activations import relu, sigmoid, softmax, activation_derivative

class Layer:
	def __init__(self, n_in, n_out, activation="sigmoid", rng=None):
		rng = rng if rng is not None else np.random.default_rng()

		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation

		if activation == "relu":
			limite = np.sqrt(6 / n_in)
		else:
			limite = np.sqrt(6 / (n_in + n_out))

		self.W = rng.uniform(-limite, limite, size=(n_out, n_in))
		self.b = np.zeros(n_out)

		self.dW = None
		self.db = None
		self.a_prev = None
		self.z = None
		self.a = None
		self.mW = np.zeros_like(self.W)#1 moment WMomentum
		self.vW = np.zeros_like(self.W)#2 moment W RMSprop
		self.mb = np.zeros_like(self.b)#1 moment b
		self.vb = np.zeros_like(self.b)#2 moment b
		self.t = 0 #compteur d etapes

	def forward(self, a_prev):
		# a_prev a une forme (N, n_in) ou (n_in,)
		self.a_prev = a_prev
		
		#produit matriceille de con : (N, n_in) @ (n_in, n_out) -> (N, n_out)
		self.z = a_prev @ self.W.T + self.b

		if self.activation == "relu":
			self.a = relu(self.z)
		elif self.activation == "sigmoid":
			self.a = sigmoid(self.z)
		elif self.activation == "tanh":
			self.a = np.tanh(self.z)
		elif self.activation == "softmax":
			self.a = softmax(self.z)
		else:
			raise ValueError(f"Wrong activation: {self.activation}")

		return self.a

	def backward(self, delta):
		# delta a une forme (N, n_out)
		N = delta.shape[0] if delta.ndim > 1 else 1

		# On stocke les gradients dans self.dW et self.db pour l'optimiseur
		self.dW = (delta.T @ self.a_prev) / N
		self.db = np.mean(delta, axis=0) if delta.ndim > 1 else delta

		# Propagation vers la couche précédente : (N, n_out) @ (n_out, n_in) -> (N, n_in)
		delta_prev = delta @ self.W

		return delta_prev


	def backward_adam(self, delta, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
		N = delta.shape[0] if delta.ndim > 1 else 1
		dW = (delta.T @ self.a_prev) / N
		db = np.mean(delta, axis=0) if delta.ndim > 1 else delta
		delta_prev = delta @ self.W
		self.t += 1
		#mise a jour premier moment
		self.mW = beta1 * self.mW + (1 - beta1) * dW
		self.mb = beta1 * self.mb + (1 - beta1) * db
		#mise a jour deuxieme moment
		self.vW = beta2 * self.vW + (1 - beta2) * (dW ** 2)
		self.vb = beta2 * self.vb + (1 - beta2) * (db ** 2)
		#correction biais
		mW_hat = self.mW / (1 - beta1 ** self.t)
		mb_hat = self.mb / (1 - beta1 ** self.t)
		vW_hat = self.vW / (1 - beta2 ** self.t)
		vb_hat = self.vb / (1 - beta2 ** self.t)
		self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
		self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

		return delta_prev