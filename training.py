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
import csv
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


def extract_data(csv_path, solution_path=None):
	try:
		if solution_path is None:
			columns = load_csv(csv_path, INFO, allow_empty_diagnosis=False)
			raw_labels = columns["Diagnosis"]
		else:
			columns = load_csv(csv_path, INFO, allow_empty_diagnosis=True)
			sol_dict = {}
			with open(solution_path, "r", encoding="utf-8") as f:
				reader = csv.reader(f)
				header = next(reader, None)
				for row_num, row in enumerate(reader, start=2):
					if len(row) != 2:
						raise ValueError(f"Invalid format in solution file '{solution_path}' at line {row_num}")
					sol_dict[row[0].strip()] = row[1].strip()
			raw_labels = []
			for i, id_val in enumerate(columns["ID"], start=1):
				id_str = str(id_val).strip()
				if id_str not in sol_dict:
					raise ValueError(f"ID '{id_str}' at row {i} not found in solution file '{solution_path}'.")
				raw_labels.append(sol_dict[id_str])
		y = []
		for i, label in enumerate(raw_labels, start=1):
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


def plot_learning_curves(history):
	try:
		epochs = range(1, len(history["loss"]) + 1)
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
		#loss curve
		ax1.plot(epochs, history["loss"], label="Train Loss", color="blue")
		ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="red", linestyle="--")
		ax1.set_title("Binary Cross-Entropy Loss")
		ax1.set_xlabel("Epochs")
		ax1.set_ylabel("Loss")
		ax1.legend()
		ax1.grid(True)
		# acc curve
		ax2.plot(epochs, history["acc"], label="Train Accuracy", color="blue")
		ax2.plot(epochs, history["val_acc"], label="Validation Accuracy", color="red", linestyle="--")
		ax2.set_title("Accuracy")
		ax2.set_xlabel("Epochs")
		ax2.set_ylabel("Accuracy")
		ax2.legend()
		ax2.grid(True)
		plt.tight_layout()
		plt.show()
	except Exception as e:
		print(f"[WARNING] Unable to display learning curves plot: {e}", file=sys.stderr)


def main():
	#parse
	parser = argparse.ArgumentParser(description="Neural Network Training Script")
	parser.add_argument("--train", type=str, required=True, help="Path to the training CSV dataset")
	parser.add_argument("--val", type=str, required=True, help="Path to the validation CSV dataset")
	parser.add_argument("--val_solution", type=str, required=True, help="Path to the solution CSV file for the validation set")
	parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
	parser.add_argument("--save_path", type=str, default="saved_model.npy", help="Path to save the trained model")
	args = parser.parse_args()
	if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
		print("[ERROR] Epochs, batch size, and learning rate must be strictly positive.", file=sys.stderr)
		sys.exit(1)

	#load Datasets
	print(f"> Loading training dataset from '{args.train}'...")
	X_train_raw, y_train = extract_data(args.train, solution_path=None)
	
	print(f"> Loading validation dataset from '{args.val}' mapped with '{args.val_solution}'...")
	X_val_raw, y_val = extract_data(args.val, solution_path=args.val_solution)

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

	#net init
	try:
		#30 inputs -> 24 hidden -> 24 hidden -> 2 outputs (Softmax)
		layer_sizes = [30, 24, 24, 2]
		activations = ["relu", "relu", "softmax"]
		net = Network(layer_sizes, activations)
	except Exception as e:
		print(f"[ERROR] Network initialization failed: {e}", file=sys.stderr)
		sys.exit(1)

	#traning
	print(f"> Starting training on {X_train.shape[0]} samples (validation: {X_val.shape[0]} samples)...")
	try:
		history = net.fit(
			X_train, y_train,
			X_val, y_val,
			epochs=args.epochs,
			batch_size=args.batch_size,
			lr=args.lr
		)
	except Exception as e:
		print(f"[ERROR] Training execution failed: {e}", file=sys.stderr)
		sys.exit(1)

	#save Model
	try:
		net.save(args.save_path)
	except Exception as e:
		print(f"[ERROR] Model saving failed: {e}", file=sys.stderr)
		sys.exit(1)

	#pplot Results
	plot_learning_curves(history)


if __name__ == "__main__":
	main()