# +------------------------------------+
# | CHAT-DISPARU // PERSONAL HEADER    |
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
# | File    : describe.py              |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 03/11/2026 14:40:34      |
# | Updated : 03/11/2026 14:40:34      |
# | END CHAT-DISPARU HEADER            |
# +------------------------------------+

import sys
import os
from lib.csv_reader import load_csv

INFO = [
	"ID",
	"Diagnosis",

	"Radius_mean",
	"Texture_mean",
	"Perimeter_mean",
	"Area_mean",
	"Smoothness_mean",
	"Compactness_mean",
	"Concavity_mean",
	"Concave_points_mean",
	"Symmetry_mean",
	"Fractal_dimension_mean",

	"Radius_se",
	"Texture_se",
	"Perimeter_se",
	"Area_se",
	"Smoothness_se",
	"Compactness_se",
	"Concavity_se",
	"Concave_points_se",
	"Symmetry_se",
	"Fractal_dimension_se",

	"Radius_worst",
	"Texture_worst",
	"Perimeter_worst",
	"Area_worst",
	"Smoothness_worst",
	"Compactness_worst",
	"Concavity_worst",
	"Concave_points_worst",
	"Symmetry_worst",
	"Fractal_dimension_worst"
]

import csv
import random


def split_dataset(data, ratio, r):
	os.makedirs("datasets", exist_ok=True)
	seed = random.SystemRandom().randint(0, 2**32 - 1) if r else 42
	rng = random.Random(seed)

	indices = list(range(len(data["ID"])))
	rng.shuffle(indices)

	split_index = int(len(indices) * ratio)

	train_indices = indices[:split_index]
	predict_indices = indices[split_index:]

	with open("datasets/data_train.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		for i in train_indices:
			writer.writerow([data[col][i] for col in INFO])

	with open("datasets/data_predict.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		for i in predict_indices:
			row = [data[col][i] for col in INFO]
			row[1] = ""
			writer.writerow(row)

	with open("datasets/solution.csv", "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["ID", "solution"])
		for i in predict_indices:
			writer.writerow([data["ID"][i], data["Diagnosis"][i]])

def main():
	args = [a for a in sys.argv[1:] if a not in ("--help", "--rand")]
	h = "--help" in sys.argv[1:]
	r = "--rand" in sys.argv[1:]
	if h:
		print("""Usage: python3 describe.py <data.csv> [--help]

Attribute information

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32) Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry
	j) fractal dimension ("coastline approximation" - 1)

The mean, standard error (SE), and worst (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.

For instance:
	Field 3  = Mean Radius
	Field 13 = Radius SE
	Field 23 = Worst Radius

All feature values are recoded with four significant digits.
""")
		sys.exit(0)

	if len(args) != 2:
		print("Usage: python3 describe.py <data.csv> [--help] [--rand] ratio", file=sys.stderr)
		sys.exit(1)

	try:
		colonnes = load_csv(args[0], INFO)
	except (ValueError, OSError, csv.Error) as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)
	try:
		ratio = float(args[1])
	except ValueError:
		print("Error: ratio must be a number", file=sys.stderr)
		sys.exit(1)

	if not 0 <= ratio <= 1:
		print("Error: ratio must be between 0 and 1", file=sys.stderr)
		sys.exit(1)
	split_dataset(colonnes, ratio, r)
	
if __name__ == "__main__":
	main()