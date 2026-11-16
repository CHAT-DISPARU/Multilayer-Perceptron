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
# | File    : optimiseur.py            |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 16/11/2026 11:54:24      |
# | Updated : 16/11/2026 11:54:24      |
# | END HEADER                         |
# +------------------------------------+
import numpy as np

class SGD:
	def __init__(self, layers, lr=0.01):
		self.layers = layers
		self.lr = lr

	def step(self):
		for layer in self.layers:
			layer.W -= self.lr * layer.dW
			layer.b -= self.lr * layer.db


class Adam:
	def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
		self.layers = layers
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.t = 0

		# L'optimiseur gère lui-même sa mémoire pour chaque couche
		self.mW = [np.zeros_like(l.W) for l in layers]
		self.vW = [np.zeros_like(l.W) for l in layers]
		self.mb = [np.zeros_like(l.b) for l in layers]
		self.vb = [np.zeros_like(l.b) for l in layers]

	def step(self):
		self.t += 1
		for i, layer in enumerate(self.layers):
			# 1er moment
			self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * layer.dW
			self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.db

			# 2nd moment
			self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * (layer.dW ** 2)
			self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.db ** 2)

			# Correction du biais
			mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
			mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
			vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
			vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

			# Mise à jour
			layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
			layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
