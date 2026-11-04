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
# | File    : activations.py           |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 04/11/2026 12:56:52      |
# | Updated : 04/11/2026 12:56:52      |
# | END HEADER                         |
# +------------------------------------+

import numpy as np

def relu(z):
	return np.maximum(0, z)

def relu_derivative(z):
	return np.where(z > 0, 1, 0)

def sigmoid(z):
	"""pas de risque overflow"""
	z_clipped = np.clip(z, -500, 500)
	return 1 / (1 + np.exp(-z_clipped))

#f(x) = 1 / (1 + e^-x)
#On pose u(x) = 1 + e^-x, d'où la dérivée u'(x) = -e^-x
#
# En utilisant la règle de dérivation (1/u)' = -u' / u² :
#   f'(x) = -(-e^-x) / (1 + e^-x)²
#   f'(x) = e^-x / (1 + e^-x)²
#
# Séparons la fraction en un produit de deux termes :
#   f'(x) = [1 / (1 + e^-x)] * [e^-x / (1 + e^-x)]
#
# Le premier terme correspond exactement à f(x). Modifions le second terme :
#   e^-x / (1 + e^-x) = (1 + e^-x - 1) / (1 + e^-x)
#                     = (1 + e^-x)/(1 + e^-x) - 1/(1 + e^-x)
#                     = 1 - [1 / (1 + e^-x)]
#                     = 1 - f(x)
#
# En réassemblant les deux morceaux :
#   f'(x) = f(x) * (1 - f(x))

def sigmoid_derivative(z):
	return sigmoid(z) * (1 - sigmoid(z))


def tanh_derivative(z):
	return 1 - np.tanh(z)**2

# exp(Aj - A(max)) / SUMj exp(Aj - A(max))
def softmax(z):
	# stabilisation on soustrait par le max
	if z.ndim == 1:
		shift_z = z - np.max(z)
		exps = np.exp(shift_z)
		return exps / np.sum(exps)
	else:
		shift_z = z - np.max(z, axis=1, keepdims=True)
		exps = np.exp(shift_z)
		return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_derivative(z):
	"""Calcule la matrice jacobienne de la fonction softmax pour un vecteur z."""
	# 1. Calculer d'abord le softmax du vecteur
	s = softmax(z)
	# 2. Transformer le vecteur s en une matrice diagonale (s_i si i==j)
	diag_s = np.diag(s)
	# 3. Calculer le produit extérieur (s_i * s_j pour tout i, j)
	outer_s = np.outer(s, s)
	# 4. Appliquer la formule : s_i * delta_ij - s_i * s_j
	return diag_s - outer_s


def activation_derivative(z, activation):
	if activation == "relu":
		return relu_derivative(z)
	elif activation == "sigmoid":
		return sigmoid_derivative(z)
	elif activation == "tanh":
		return tanh_derivative(z)
	elif activation == "softmax":
		return softmax_derivative(z)
	else:
		raise ValueError(f"Activation inconnue: {activation}")