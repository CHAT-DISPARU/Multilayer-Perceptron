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
# | File    : losses.py                |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 04/11/2026 12:57:04      |
# | Updated : 04/11/2026 12:57:04      |
# | END HEADER                         |
# +------------------------------------+

import numpy as np

def binary_cross_entropy(y_pred, y_true, epsilon=1e-15):
	#securité on limitevaleurs entre epsilon et 1-epsilon
	#eviter log(0) qui produit une erreur -inf / NaN
	y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
	# si les prédictions sont en Softmax avec 2 sorties (N, 2) [Maligne, Bénigne]
	if y_pred.ndim == 2 and y_pred.shape[1] == 2:
		p = y_pred[:, 1]
		y = y_true[:, 1]
	else:
		p = y_pred
		y = y_true
	# perte moyenne
	loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
	return loss

def compute_accuracy(y_pred, y_true):
    # Cas Softmax avec 2 sorties One-Hot (ex: [0, 1] vs [1, 0])
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
    # Cas Sigmoïde avec 1 seule sortie probabilité (0 à 1)
    else:
        pred_labels = (y_pred >= 0.5).astype(int)
        true_labels = y_true

    #moyenne des égalités fraction de prédictions correctes 0 1
    return np.mean(pred_labels == true_labels)