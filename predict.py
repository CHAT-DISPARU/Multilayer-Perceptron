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
# | Created : 22/11/2026 16:51:56      |
# | Updated : 22/11/2026 16:51:56      |
# | END HEADER                         |
# +------------------------------------+




import argparse
import csv
import sys
import numpy as np
from mlp.columns import INFO
from mlp.csv_reader import load_csv
from mlp.losses import LOSSES, compute_accuracy, confusion_counts, get_loss, precision_recall_f1
from mlp.network import Network


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


def load_model(model_path):
	"""recharge le reseau (poids + standardisation du train)"""
	return Network.load(model_path)


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
	parser.add_argument("--output", default="predictions.csv", help="Path of the predictions file")
	parser.add_argument("--loss", choices=list(LOSSES), default=None,
			help="Loss to report (default: the one used to train the model)")
	args = parser.parse_args()

	try:
		net = load_model(args.model)
		ids, X_raw, y_true = load_dataset(args.dataset)
		if net.mean is None or net.std is None:
			raise ValueError("Model has no normalization parameters: retrain it with the current training.py")
		if X_raw.shape[1] != net.mean.shape[0] or X_raw.shape[1] != net.layers[0].n_in:
			raise ValueError(
				f"Dataset has {X_raw.shape[1]} features, "
				f"model expects {net.layers[0].n_in} (normalization: {net.mean.shape[0]})"
			)
	except FileNotFoundError as e:
		print(f"[ERROR] File not found: '{e.filename}'", file=sys.stderr)
		sys.exit(1)
	except (ValueError, KeyError, OSError, csv.Error) as e:
		print(f"[ERROR] {e}", file=sys.stderr)
		sys.exit(1)

	X = net.normalize(X_raw)
	y_pred = net.predict(X)

	loss_name = args.loss or net.loss
	loss = get_loss(loss_name)(y_pred, y_true)
	acc = compute_accuracy(y_pred, y_true)
	tp, fp, fn, tn = confusion_counts(y_pred, y_true)
	precision, recall, f1 = precision_recall_f1(y_pred, y_true)

	print(f"> {X.shape[0]} samples loaded from '{args.dataset}'")
	print(f"> {loss_name}: {loss:.4f}")
	print(f"> accuracy: {acc * 100:.2f} %")
	print(f"> malignant detected: {tp}  missed: {fn}  false alarms: {fp}  correct benign: {tn}")
	print(f"> precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")
	save_predictions(args.output, ids, y_pred)
	print(f"> predictions saved to '{args.output}'")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n> interrupted", file=sys.stderr)
		sys.exit(130)

