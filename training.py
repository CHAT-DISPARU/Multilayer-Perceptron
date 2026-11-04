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
		
		num_plots = len(histories)
		fig, axes = plt.subplots(num_plots, 2, figsize=(14, 6 * num_plots))
		
		if num_plots == 1:
			axes = [axes]
			
		for (act_name, history), (ax1, ax2) in zip(histories.items(), axes):
			epochs = range(1, len(history["loss"]) + 1)
			total_epochs = len(epochs)

			final_loss = history["loss"][-1]
			final_val_loss = history["val_loss"][-1]
			final_acc = history["acc"][-1]
			final_val_acc = history["val_acc"][-1]

			color_train = "cyan"
			color_val = "orange"

			ax1.plot(epochs, history["loss"], label="Train Loss", color=color_train)
			ax1.plot(epochs, history["val_loss"], label="Validation Loss", color=color_val, linestyle="--")

			title_loss = f"Loss ({act_name.upper()}) - Epoch {total_epochs}/{total_epochs}\nLoss: {final_loss:.4f} | Val_Loss: {final_val_loss:.4f}"
			ax1.set_title(title_loss, pad=15, fontsize=12)
			ax1.set_xlabel("Epochs")
			ax1.set_ylabel("Loss")
			ax1.legend()
			ax1.grid(True, alpha=0.3)

			ax2.plot(epochs, history["acc"], label="Train Accuracy", color=color_train)
			ax2.plot(epochs, history["val_acc"], label="Validation Accuracy", color=color_val, linestyle="--")
			
			title_acc = f"Accuracy ({act_name.upper()}) - Epoch {total_epochs}/{total_epochs}\nAcc: {final_acc:.4f} | Val_Acc: {final_val_acc:.4f}"
			ax2.set_title(title_acc, pad=15, fontsize=12)
			ax2.set_xlabel("Epochs")
			ax2.set_ylabel("Accuracy")
			ax2.legend()
			ax2.grid(True, alpha=0.3)
		plt.tight_layout(pad=3.0, h_pad=4.0)
		plt.show()
	except Exception as e:
		print(f"[WARNING] Unable to display learning curves plot: {e}", file=sys.stderr)


def main():
	#parse
	parser = argparse.ArgumentParser(description="Neural Network Training Script")
	parser.add_argument("--train", type=str, required=True, help="Path to the training CSV dataset")
	parser.add_argument("--val", type=str, required=True, help="Path to the validation CSV dataset")
	parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
	parser.add_argument("--save_path", type=str, default="saved_model.npy", help="Path to save the trained model")
	parser.add_argument("--all", action="store_true", help="all actuivation function test")
	args = parser.parse_args()
	if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
		print("[ERROR] Epochs, batch size, and learning rate must be strictly positive.", file=sys.stderr)
		sys.exit(1)

	#load Datasets
	print(f"> Loading training dataset from '{args.train}'...")
	X_train_raw, y_train = extract_data(args.train)
	
	print(f"> Loading validation dataset from '{args.val}'...")
	X_val_raw, y_val = extract_data(args.val)

	#standardization
	print("> Standardizing features...")
	try:
		mean = np.mean(X_train_raw, axis=0)
		std = np.std(X_train_raw, axis=0)
		std[std == 0] = 1e-8  #prevent /0

		X_train = (X_train_raw - mean) / std
		X_val = (X_val_raw - mean) / std
		
		# Save normalization for predict.py
		np.savez("normalization_params.npz", mean=mean, std=std)
	except Exception as e:
		print(f"[ERROR] Failed to standardize data or save parameters: {e}", file=sys.stderr)
		sys.exit(1)

	print(f"> Starting training on {X_train.shape[0]} samples (validation: {X_val.shape[0]} samples)...")
	
	# Détermination des activations à tester
	activations_to_test = ["relu", "sigmoid", "tanh"] if args.all else ["relu"]
	histories = {}

	for act in activations_to_test:
		print(f"\n--- Entraînement avec l'activation : {act.upper()} ---")
		#net init
		try:
			#30 inputs -> 24 hidden -> 24 hidden -> 2 outputs (Softmax)
			layer_sizes = [30, 24, 24, 2]
			activations = [act, act, "softmax"]
			net = Network(layer_sizes, activations)
		except Exception as e:
			print(f"[ERROR] Network initialization failed: {e}", file=sys.stderr)
			sys.exit(1)

		#traning
		try:
			history = net.fit(
				X_train, y_train,
				X_val, y_val,
				epochs=args.epochs,
				batch_size=args.batch_size,
				lr=args.lr
			)
			histories[act] = history
		except Exception as e:
			print(f"[ERROR] Training execution failed: {e}", file=sys.stderr)
			sys.exit(1)

		#save Model
		if act == "relu":
			try:
				net.save(args.save_path)
			except Exception as e:
				print(f"[ERROR] Model saving failed: {e}", file=sys.stderr)
				sys.exit(1)

	#plot Results
	plot_learning_curves(histories)


if __name__ == "__main__":
	main()