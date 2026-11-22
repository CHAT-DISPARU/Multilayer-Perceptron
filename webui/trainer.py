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
# | File    : trainer.py               |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 21/11/2026 21:37:25      |
# | Updated : 21/11/2026 21:37:25      |
# | END HEADER                         |
# +------------------------------------+

import io
import json
import threading
from contextlib import redirect_stdout

import numpy as np

from mlp.losses import LOSSES as LOSS_FUNCS, compute_accuracy, confusion_counts, get_loss, precision_recall_f1
from mlp.network import Network
from mlp.optimiseur import Adam, SGD
from predict import load_dataset, save_predictions

METRICS = ["loss", "val_loss", "acc", "val_acc", "precision", "val_precision",
		   "recall", "val_recall", "f1", "val_f1"]
ACTS = ("relu", "sigmoid", "tanh")
OPTS = ("adam", "sgd")
LOSSES = tuple(LOSS_FUNCS)

_lock = threading.Lock()
_stop = threading.Event()
_thread = None
_net = None
_state = {"status": "idle"}


def _log(line):
	with _lock:
		_state["log"].append(line)
	try:
		print(line, flush=True)
	except Exception:
		pass


def _epoch_line(ep, total, h):
	return (f"epoch {ep:02d}/{total} - loss: {h['loss'][0]:.4f} - val_loss: {h['val_loss'][0]:.4f}"
			f" - acc: {h['acc'][0]:.4f} - val_acc: {h['val_acc'][0]:.4f}"
			f" - val_precision: {h['val_precision'][0]:.4f} - val_recall: {h['val_recall'][0]:.4f}"
			f" - val_f1: {h['val_f1'][0]:.4f}")


def _save_history(path, cfg, history, best_epoch):
	"""save_history() training.py"""
	data = {
		"optimizer": cfg["opt"], "activation": cfg["act"], "learning_rate": cfg["lr"],
		"layers": cfg["layers"], "batch_size": cfg["batch"], "epochs_requested": cfg["epochs"],
		"epochs_run": len(history["loss"]), "loss": cfg["loss"], "patience": cfg["patience"],
		"best_epoch": best_epoch, "history": history,
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, default=float)


def _weights(net):
	return [{"W": l.W.tolist(), "b": l.b.tolist(), "act": l.activation} for l in net.layers]


def _check(cfg):
	layers = [int(n) for n in cfg.get("layers", [24, 24])]
	if len(layers) < 2:
		raise ValueError("At least 2 hidden layers are required")
	if any(n <= 0 for n in layers):
		raise ValueError("Layer sizes must be strictly positive integers")
	act, opt, loss = cfg.get("act", "relu"), cfg.get("opt", "adam"), cfg.get("loss", LOSSES[0])
	if act not in ACTS or opt not in OPTS or loss not in LOSSES:
		raise ValueError("Unknown activation, optimizer or loss")
	epochs, batch = int(cfg.get("epochs", 70)), int(cfg.get("batch", 16))
	patience = int(cfg.get("patience") or 0)
	lr = cfg.get("lr")
	lr = None if lr in (None, "") else float(lr)
	if epochs <= 0 or batch <= 0 or (lr is not None and lr <= 0):
		raise ValueError("Epochs, batch size, and learning rate must be strictly positive")
	if patience < 0:
		raise ValueError("Patience must be >= 0")
	if lr is None:
		lr = 0.001 if opt == "adam" else 0.01
	return dict(layers=layers, act=act, opt=opt, loss=loss, epochs=epochs,
				batch=batch, patience=patience, lr=lr)


def start(payload):
	"""valide charge les donnees  lance le thread"""
	global _thread
	with _lock:
		if _state.get("status") == "running":
			raise ValueError("A training is already running")
	cfg = _check(payload)
	train_path = (payload.get("train") or "datasets/data_train.csv").strip()
	val_path = (payload.get("val") or "datasets/data_predict.csv").strip()
	try:
		_, X_raw, y = load_dataset(train_path)
		_, Xv_raw, yv = load_dataset(val_path)
	except FileNotFoundError as e:
		raise ValueError(f"File not found: '{e.filename}' (split the data first)")
	if len(X_raw) == 0 or len(Xv_raw) == 0:
		raise ValueError("Train and validation sets must not be empty")

	mean = np.mean(X_raw, axis=0)
	std = np.std(X_raw, axis=0)
	std[std == 0] = 1e-8
	net = Network([X_raw.shape[1]] + cfg["layers"] + [2], [cfg["act"]] * len(cfg["layers"]) + ["softmax"])
	net.set_normalization(mean, std)
	net.loss = cfg["loss"]
	opt_cls = Adam if cfg["opt"] == "adam" else SGD
	optimizer = opt_cls(net.layers, lr=cfg["lr"])

	_stop.clear()
	with _lock:
		_state.clear()
		_state.update(status="running", epoch=0, epochs=cfg["epochs"], config=cfg,
					  history={k: [] for k in METRICS}, best_epoch=None, early_stopped=False,
					  error=None, model_file=None, weights=_weights(net),
					  n_train=len(X_raw), n_val=len(Xv_raw), log=[], history_file=None)
	_log(f"x_train shape : {X_raw.shape}")
	_log(f"x_valid shape : {Xv_raw.shape}")
	_log(f"--- Entraînement : {cfg['opt'].upper()} (lr={cfg['lr']}) | Activation : {cfg['act'].upper()} ---")
	_thread = threading.Thread(
		target=_run, args=(net, optimizer, cfg, (net.normalize(X_raw), y), (net.normalize(Xv_raw), yv)), daemon=True)
	_thread.start()
	return {"started": True}


def _run(net, optimizer, cfg, train, val):
	global _net
	X, y = train
	Xv, yv = val
	best, best_w, best_epoch, wait = float("inf"), None, None, 0
	try:
		for ep in range(1, cfg["epochs"] + 1):
			if _stop.is_set():
				break
			with redirect_stdout(io.StringIO()):
				h = net.fit(X, y, Xv, yv, epochs=1, batch_size=cfg["batch"], optimizer=optimizer, patience=0, loss=cfg["loss"])
			if not (np.isfinite(h["loss"][0]) and np.isfinite(h["val_loss"][0])):
				raise ValueError("Loss diverged (NaN/inf): lower the learning rate")
			with _lock:
				for k in METRICS:
					_state["history"][k].append(float(h[k][0]))
				_state["epoch"] = ep
				_state["weights"] = _weights(net)
			_log(_epoch_line(ep, cfg["epochs"], h))
			if cfg["patience"] > 0:
				vl = h["val_loss"][0]
				if vl < best:
					best, best_epoch, wait = vl, ep, 0
					best_w = [(l.W.copy(), l.b.copy()) for l in net.layers]
				else:
					wait += 1
					if wait >= cfg["patience"]:
						with _lock:
							_state["early_stopped"] = True
						_log(f"early stopping at epoch {ep} (best epoch: {best_epoch}, val_loss: {best:.4f})")
						break
		if best_w is not None:# restore meilleur comme training.py
			for layer, (W, b) in zip(net.layers, best_w):
				layer.W, layer.b = W, b
			_log(f"> best weights restored (epoch {best_epoch})")
		path = f"saved_model_{cfg['opt']}_{cfg['act']}.npy"
		with redirect_stdout(io.StringIO()):
			net.save(path)
		_log(f"> saving model '{path}' to disk...")
		hist_path = f"saved_model_{cfg['opt']}_{cfg['act']}_history.json"
		with _lock:
			history = {k: list(v) for k, v in _state["history"].items()}
		_save_history(hist_path, cfg, history, best_epoch)
		_log(f"> history saved to '{hist_path}'")
		_net = net
		with _lock:
			_state["best_epoch"] = best_epoch
			_state["weights"] = _weights(net)
			_state["model_file"] = path
			_state["history_file"] = hist_path
			_state["status"] = "stopped" if _stop.is_set() else "done"
	except Exception as e:#pas mort silence
		with _lock:
			_state["status"] = "error"
			_state["error"] = str(e)


def status(_payload=None):
	with _lock:
		return dict(_state)


def stop(_payload=None):
	_stop.set()
	return {"stopping": True}


def predict(payload):
	"""meme predict.py"""
	if _net is None:
		raise ValueError("Train a model first")
	if _state.get("status") == "running":
		raise ValueError("Wait for the training to finish")
	path = (payload.get("dataset") or "datasets/data_predict.csv").strip()
	try:
		ids, X_raw, y_true = load_dataset(path)
	except FileNotFoundError:
		raise ValueError(f"File not found: '{path}'")
	if X_raw.shape[1] != _net.mean.shape[0]:
		raise ValueError(f"Dataset has {X_raw.shape[1]} features, model expects {_net.mean.shape[0]}")
	y_pred = _net.predict(_net.normalize(X_raw))
	tp, fp, fn, tn = confusion_counts(y_pred, y_true)
	precision, recall, f1 = precision_recall_f1(y_pred, y_true)
	save_predictions("predictions.csv", ids, y_pred)
	labels = np.argmax(y_pred, axis=1)
	truth = np.argmax(y_true, axis=1)
	rows = [{"id": str(i), "true": "M" if t == 0 else "B", "pred": "M" if p == 0 else "B",
			 "p_m": float(q[0]), "p_b": float(q[1])}
			for i, t, p, q in zip(ids, truth, labels, y_pred)]
	return {"samples": int(X_raw.shape[0]), "dataset": path, "output": "predictions.csv",
			"loss": float(get_loss(_net.loss)(y_pred, y_true)), "loss_name": _net.loss,
			"accuracy": float(compute_accuracy(y_pred, y_true)),
			"tp": tp, "fp": fp, "fn": fn, "tn": tn,
			"precision": float(precision), "recall": float(recall), "f1": float(f1),
			"rows": rows}
