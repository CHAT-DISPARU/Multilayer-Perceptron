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
# | File    : api.py                   |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 21/11/2026 21:37:44      |
# | Updated : 21/11/2026 21:37:44      |
# | END HEADER                         |
# +------------------------------------+

from mlp.columns import INFO
from mlp.csv_reader import load_csv
from split import split_dataset
from webui import trainer


def _read(path):
	path = (path or "").strip()
	if not path:
		raise ValueError("Path is empty")
	try:
		return load_csv(path, INFO)
	except FileNotFoundError:
		raise ValueError(f"File not found: '{path}'")


def inspect(payload):
	"""sert à  la barre coulissante"""
	diag = _read(payload.get("path"))["Diagnosis"]
	return {"total": len(diag), "malignant": diag.count("M"), "benign": diag.count("B")}


def split(payload):
	"""appelle vrai split_dataset split.py"""
	data = _read(payload.get("path"))
	ratio = float(payload.get("ratio", 0.8))
	if not 0 < ratio < 1:
		raise ValueError("ratio must be strictly between 0 and 1")
	split_dataset(data, ratio, bool(payload.get("rand")))
	total = len(data["ID"])
	n_train = int(total * ratio)
	return {"train": n_train, "test": total - n_train,
			"train_file": "datasets/data_train.csv", "test_file": "datasets/data_predict.csv"}


ROUTES = {
	"/api/inspect": inspect,
	"/api/split": split,
	"/api/train": trainer.start,
	"/api/status": trainer.status,
	"/api/stop": trainer.stop,
	"/api/predict": trainer.predict,
}