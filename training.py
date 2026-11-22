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
# | File    : training.py              |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 04/11/2026 17:27:46      |
# | Updated : 04/11/2026 17:27:46      |
# | END HEADER                         |
# +------------------------------------+

import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

from mlp.columns import INFO
from mlp.csv_reader import load_csv
from mlp.losses import binary_cross_entropy, compute_accuracy
from mlp.network import Network
from mlp.optimiseur import Adam, SGD



def extract_data(csv_path):
	try:
		columns = load_csv(csv_path, INFO)
		y = []
		for i, label in enumerate(columns["Diagnosis"], start=1):
			if label == "M":
				y.append([1, 0])
			elif label == "B":
				y.append([0, 1])
			else:
				raise ValueError(f"Unknown label '{label}' at row {i}. Expected 'M' or 'B'.")
		feature_names = INFO[2:]
		features_list = [columns[name] for name in feature_names]
		X = np.column_stack(features_list).astype(float)
		
		if X.shape[0] == 0:
			raise ValueError(f"Dataset '{csv_path}' is empty.")
			
		return X, np.array(y)

	except FileNotFoundError as e:
		print(f"[ERROR] File not found: '{e.filename}'", file=sys.stderr)
		sys.exit(1)
	except Exception as e:
		print(f"[ERROR] Failed to process data: {e}", file=sys.stderr)
		sys.exit(1)


def plot_learning_curves(histories):
	try:
		plt.style.use('dark_background')
		
		num_opts = len(histories)
		fig, axes = plt.subplots(num_opts, 2, figsize=(15, 6 * num_opts))
		
		# Si 1 optimiseur axes est 1D transforme en 2D
		if num_opts == 1:
			axes = [axes]

		colors = {
			"relu": "cyan",
			"sigmoid": "orange",
			"tanh": "lime"
		}
			
		for (opt_name, act_dict), (ax1, ax2) in zip(histories.items(), axes):
			for act_name, history in act_dict.items():
				epochs = range(1, len(history["loss"]) + 1)
				c = colors.get(act_name, "white")

				best = history.get("best_epoch")
				idx = best - 1 if best else -1
				final_val_loss = history["val_loss"][idx]
				final_val_acc = history["val_acc"][idx]

				#loss
				ax1.plot(epochs, history["loss"], label=f"{act_name.upper()} Train", color=c, alpha=0.8)
				ax1.plot(epochs, history["val_loss"], label=f"{act_name.upper()} Val ({final_val_loss:.4f})", color=c, linestyle="--")
				#acc
				ax2.plot(epochs, history["acc"], label=f"{act_name.upper()} Train", color=c, alpha=0.8)
				ax2.plot(epochs, history["val_acc"], label=f"{act_name.upper()} Val ({final_val_acc:.4f})", color=c, linestyle="--")

			ax1.set_title(f"Loss Comparison - Optimizer: {opt_name.upper()}", pad=12, fontsize=13, fontweight="bold")
			ax1.set_xlabel("Epochs")
			ax1.set_ylabel("Loss")
			ax1.legend(loc="upper right", fontsize=9)
			ax1.grid(True, alpha=0.3)

			ax2.set_title(f"Accuracy Comparison - Optimizer: {opt_name.upper()}", pad=12, fontsize=13, fontweight="bold")
			ax2.set_xlabel("Epochs")
			ax2.set_ylabel("Accuracy")
			ax2.legend(loc="lower right", fontsize=9)
			ax2.grid(True, alpha=0.3)

		plt.tight_layout(pad=3.0, h_pad=4.0)
		plt.show()
	except Exception as e:
		print(f"[WARNING] Unable to display learning curves plot: {e}", file=sys.stderr)

def save_history(path, history, opt_name, act, lr, args):
	data = {
		"optimizer": opt_name,
		"activation": act,
		"learning_rate": lr,
		"layers": args.layer,
		"batch_size": args.batch_size,
		"epochs_requested": args.epochs,
		"epochs_run": len(history["loss"]),
		"loss": args.loss,
		"patience": args.patience,
		"best_epoch": history.get("best_epoch"),
		"history": {k: v for k, v in history.items() if k != "best_epoch"},
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, default=float)
	print(f"> history saved to '{path}'")

def main():
	# Parse
	parser = argparse.ArgumentParser(description="Neural Network Training Script")
	parser.add_argument("--train", type=str, required=True, help="Path to the training CSV dataset")
	parser.add_argument("--val", type=str, required=True, help="Path to the validation CSV dataset")
	parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
	parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=None,
					help="Learning rate (default: 0.001 for Adam, 0.01 for SGD)")
	parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="Optimizer to use (adam or sgd)")
	parser.add_argument("--loss", type=str, default="categoricalCrossentropy",
			choices=["categoricalCrossentropy", "binaryCrossentropy"],
			help="Loss function (equivalent with a 2-output softmax)")
	parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"],
					help="Hidden layers activation (ignored with --all)")
	parser.add_argument("--save_path", type=str, default="saved_model", help="Path to save the trained model")
	parser.add_argument("--all", action="store_true", help="Compare all activations (ReLU, Sigmoid, Tanh) for both SGD and Adam")
	parser.add_argument("--layer", type=int, nargs="+", default=[24, 24],
					help="Hidden layer sizes, e.g. --layer 24 24 24 (at least 2)")
	parser.add_argument("--patience", type=int, default=0,
					help="Early stopping: stop after N epochs without val_loss improvement (0 = disabled)")
	args = parser.parse_args()

	if args.epochs <= 0 or args.batch_size <= 0 or (args.lr is not None and args.lr <= 0):
		print("[ERROR] Epochs, batch size, and learning rate must be strictly positive.", file=sys.stderr)
		sys.exit(1)
	if len(args.layer) < 2:
		print("[ERROR] At least 2 hidden layers are required.", file=sys.stderr)
		sys.exit(1)
	if any(n <= 0 for n in args.layer):
		print("[ERROR] Layer sizes must be strictly positive integers.", file=sys.stderr)
		sys.exit(1)
	if args.patience < 0:
		print("[ERROR] Patience must be >= 0.", file=sys.stderr)
		sys.exit(1)
	#load data
	print(f"> Loading training dataset from '{args.train}'...")
	X_train_raw, y_train = extract_data(args.train)
	
	print(f"> Loading validation dataset from '{args.val}'...")
	X_val_raw, y_val = extract_data(args.val)

	#standardization
	print("> Standardizing features...")
	try:
		mean = np.mean(X_train_raw, axis=0)
		std = np.std(X_train_raw, axis=0)
		std[std == 0] = 1e-8

		X_train = (X_train_raw - mean) / std
		X_val = (X_val_raw - mean) / std
	except Exception as e:
		print(f"[ERROR] Failed to standardize data: {e}", file=sys.stderr)
		sys.exit(1)

	print(f"x_train shape : {X_train.shape}")
	print(f"x_valid shape : {X_val.shape}")
	
	#def configurations a lancer
	optimizers_to_test = ["sgd", "adam"] if args.all else [args.optimizer]
	activations_to_test = ["relu", "sigmoid", "tanh"] if args.all else [args.activation]
	
	# Dic imbrique histories[opt][act] = history
	histories = {}

	for opt_name in optimizers_to_test:
		histories[opt_name] = {}
		current_lr = args.lr if args.lr is not None else (0.001 if opt_name == "adam" else 0.01)

		for act in activations_to_test:
			print(f"\n--- Entraînement : {opt_name.upper()} (lr={current_lr}) | Activation : {act.upper()} ---")
			
			#network init
			try:
				layer_sizes = [X_train.shape[1]] + args.layer + [2]
				activations = [act] * len(args.layer) + ["softmax"]
				net = Network(layer_sizes, activations)
				net.set_normalization(mean, std)
				if opt_name == "adam":
					optimizer = Adam(net.layers, lr=current_lr)
				else:
					optimizer = SGD(net.layers, lr=current_lr)
			except Exception as e:
				print(f"[ERROR] Network initialization failed: {e}", file=sys.stderr)
				sys.exit(1)

			#train
			try:
				history = net.fit(
					X_train, y_train,
					X_val, y_val,
					epochs=args.epochs,
					batch_size=args.batch_size,
					optimizer=optimizer,
					patience=args.patience
				)
				histories[opt_name][act] = history
			except Exception as e:
				print(f"[ERROR] Training execution failed: {e}", file=sys.stderr)
				sys.exit(1)

			#save
			try:
				net.save(f"{args.save_path}_{opt_name}_{act}.npy")
			except Exception as e:
				print(f"[ERROR] Model saving failed: {e}", file=sys.stderr)
				sys.exit(1)
			save_history(f"{args.save_path}_{opt_name}_{act}_history.json",
				history, opt_name, act, current_lr, args)

	#plot result
	plot_learning_curves(histories)


if __name__ == "__main__":
	main()
