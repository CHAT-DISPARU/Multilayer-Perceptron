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
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

from mlp.csv_reader import load_csv
from mlp.losses import binary_cross_entropy, compute_accuracy
from mlp.network import Network
from mlp.optimiseur import Adam, SGD

INFO = [
	"ID",
	"Diagnosis",
	"Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Smoothness_mean",
	"Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean", "Fractal_dimension_mean",
	"Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se",
	"Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se", "Fractal_dimension_se",
	"Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", "Smoothness_worst",
	"Compactness_worst", "Concavity_worst", "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst"
]


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
		
		# Si un seul optimiseur est testé, axes est 1D, on le transforme en 2D
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

				final_val_loss = history["val_loss"][-1]
				final_val_acc = history["val_acc"][-1]

				# --- Graphique 1 : LOSS (Superposition) ---
				ax1.plot(epochs, history["loss"], label=f"{act_name.upper()} Train", color=c, alpha=0.8)
				ax1.plot(epochs, history["val_loss"], label=f"{act_name.upper()} Val ({final_val_loss:.4f})", color=c, linestyle="--")

				# --- Graphique 2 : ACCURACY (Superposition) ---
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


def main():
	# Parse
	parser = argparse.ArgumentParser(description="Neural Network Training Script")
	parser.add_argument("--train", type=str, required=True, help="Path to the training CSV dataset")
	parser.add_argument("--val", type=str, required=True, help="Path to the validation CSV dataset")
	parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
	parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 0.001 for Adam, 0.01 for SGD)")
	parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use (adam or sgd)")
	parser.add_argument("--save_path", type=str, default="saved_model", help="Path to save the trained model")
	parser.add_argument("--all", action="store_true", help="Compare all activations (ReLU, Sigmoid, Tanh) for both SGD and Adam")
	args = parser.parse_args()

	if args.epochs <= 0 or args.batch_size <= 0 or (args.lr is not None and args.lr <= 0):
		print("[ERROR] Epochs, batch size, and learning rate must be strictly positive.", file=sys.stderr)
		sys.exit(1)

	# Load Datasets
	print(f"> Loading training dataset from '{args.train}'...")
	X_train_raw, y_train = extract_data(args.train)
	
	print(f"> Loading validation dataset from '{args.val}'...")
	X_val_raw, y_val = extract_data(args.val)

	# Standardization
	print("> Standardizing features...")
	try:
		mean = np.mean(X_train_raw, axis=0)
		std = np.std(X_train_raw, axis=0)
		std[std == 0] = 1e-8

		X_train = (X_train_raw - mean) / std
		X_val = (X_val_raw - mean) / std
		
		np.savez("normalization_params.npz", mean=mean, std=std)
	except Exception as e:
		print(f"[ERROR] Failed to standardize data or save parameters: {e}", file=sys.stderr)
		sys.exit(1)

	print(f"> Starting training on {X_train.shape} samples (validation: {X_val.shape} samples)...")
	
	# Définition des configurations à lancer
	optimizers_to_test = ["sgd", "adam"] if args.all else [args.optimizer]
	activations_to_test = ["relu", "sigmoid", "tanh"] if args.all else ["relu"]
	
	# Dictionnaire imbriqué : histories[opt][act] = history
	histories = {}

	for opt_name in optimizers_to_test:
		histories[opt_name] = {}
		# Utilise le lr fourni par l'utilisateur, sinon un défaut adapté à chaque optimiseur
		current_lr = args.lr if args.lr is not None else (0.001 if opt_name == "adam" else 0.01)

		for act in activations_to_test:
			print(f"\n--- Entraînement : {opt_name.upper()} (lr={current_lr}) | Activation : {act.upper()} ---")
			
			# Net init
			try:
				layer_sizes = [30, 24, 24, 2]
				activations = [act, act, "softmax"]
				net = Network(layer_sizes, activations)
				
				if opt_name == "adam":
					optimizer = Adam(net.layers, lr=current_lr)
				else:
					optimizer = SGD(net.layers, lr=current_lr)
			except Exception as e:
				print(f"[ERROR] Network initialization failed: {e}", file=sys.stderr)
				sys.exit(1)

			# Training
			try:
				history = net.fit(
					X_train, y_train,
					X_val, y_val,
					epochs=args.epochs,
					batch_size=args.batch_size,
					optimizer=optimizer
				)
				histories[opt_name][act] = history
			except Exception as e:
				print(f"[ERROR] Training execution failed: {e}", file=sys.stderr)
				sys.exit(1)

			# Save Model
			try:
				net.save(f"{args.save_path}_{opt_name}_{act}.npy")
			except Exception as e:
				print(f"[ERROR] Model saving failed: {e}", file=sys.stderr)
				sys.exit(1)

	# Plot Results
	plot_learning_curves(histories)


if __name__ == "__main__":
	main()
