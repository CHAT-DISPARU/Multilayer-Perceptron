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
# | File    : predict.py               |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 04/11/2026 23:13:17      |
# | Updated : 04/11/2026 23:13:17      |
# | END HEADER                         |
# +------------------------------------+



import argparse
import csv
import sys
import numpy as np

from mlp.csv_reader import load_csv
from mlp.losses import binary_cross_entropy, compute_accuracy
from mlp.network import Network
from training import INFO


def load_dataset(path):
	"""renvoie (ids, X, y) ; y en one-hot : M = [1, 0], B = [0, 1]"""
	columns = load_csv(path, INFO)
	y = []
	for i, label in enumerate(columns["Diagnosis"], start=1):
		if label == "M":
			y.append([1, 0])
		elif label == "B":
			y.append([0, 1])
		else:
			raise ValueError(f"Unknown label '{label}' at row {i}. Expected 'M' or 'B'.")
	X = np.column_stack([columns[name] for name in INFO[2:]]).astype(float)
	return columns["ID"], X, np.array(y)


def load_model(model_path, norm_path):
	"""recharge le reseau et lstandardisation du train"""
	net = Network.load(model_path)
	params = np.load(norm_path)
	return net, params["mean"], params["std"]


def confusion(y_pred, y_true):
	"""comptes avec M comme classe positive"""
	pred_m = np.argmax(y_pred, axis=1) == 0
	true_m = np.argmax(y_true, axis=1) == 0
	tp = int(np.sum(pred_m & true_m))
	fp = int(np.sum(pred_m & ~true_m))
	fn = int(np.sum(~pred_m & true_m))
	tn = int(np.sum(~pred_m & ~true_m))
	return tp, fp, fn, tn


def save_predictions(path, ids, y_pred):
	labels = np.argmax(y_pred, axis=1)
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["ID", "Prediction", "P_M", "P_B"])
		for id_val, label, p in zip(ids, labels, y_pred):
			writer.writerow([id_val, "M" if label == 0 else "B", f"{p[0]:.6f}", f"{p[1]:.6f}"])


def main():
	parser = argparse.ArgumentParser(description="Neural Network Prediction Script")
	parser.add_argument("--dataset", required=True, help="CSV dataset (same format as the training data)")
	parser.add_argument("--model", required=True, help="Path to the saved model")
	parser.add_argument("--norm", required=True, help="Path to the normalization parameters")
	parser.add_argument("--output", default="predictions.csv", help="Path of the predictions file")
	args = parser.parse_args()

	try:
		net, mean, std = load_model(args.model, args.norm)
		ids, X_raw, y_true = load_dataset(args.dataset)
		if X_raw.shape[1] != mean.shape[0] or X_raw.shape[1] != net.layers[0].n_in:
			raise ValueError(
				f"Dataset has {X_raw.shape[1]} features, "
				f"model expects {net.layers[0].n_in} (normalization: {mean.shape[0]})"
			)
	except FileNotFoundError as e:
		print(f"[ERROR] File not found: '{e.filename}'", file=sys.stderr)
		sys.exit(1)
	except (ValueError, KeyError, OSError, csv.Error) as e:
		print(f"[ERROR] {e}", file=sys.stderr)
		sys.exit(1)

	X = (X_raw - mean) / std
	y_pred = net.predict(X)

	loss = binary_cross_entropy(y_pred, y_true)
	acc = compute_accuracy(y_pred, y_true)
	tp, fp, fn, tn = confusion(y_pred, y_true)

	print(f"> {X.shape[0]} samples loaded from '{args.dataset}'")
	print(f"> binary cross-entropy: {loss:.4f}")
	print(f"> accuracy: {acc * 100:.2f} %")
	print(f"> malignant detected: {tp}  missed: {fn}  false alarms: {fp}  correct benign: {tn}")

	save_predictions(args.output, ids, y_pred)
	print(f"> predictions saved to '{args.output}'")


if __name__ == "__main__":
	main()
