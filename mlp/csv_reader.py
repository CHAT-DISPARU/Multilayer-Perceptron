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
# | File    : csv_reader.py            |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 03/11/2026 15:24:54      |
# | Updated : 03/11/2026 15:24:54      |
# | END CHAT-DISPARU HEADER            |
# +------------------------------------+

import csv


def load_csv(path, info, allow_empty_diagnosis=False):
	colonnes = {nom: [] for nom in info}

	with open(path, newline="", encoding="utf-8") as f:
		lecteur = csv.reader(f)
		for num_ligne, ligne in enumerate(lecteur, start=1):
			if len(ligne) != len(info):
				raise ValueError(f"Invalid number of columns at line {num_ligne}")
			for nom, cellule in zip(info, ligne):
				cellule = cellule.strip()
				if not cellule:
					raise ValueError(f"Missing value at line {num_ligne}: {nom}")

					raise ValueError(f"Missing value at line {num_ligne}: {nom}")
				if cellule.lower() == "null":
					raise ValueError(f"Null value at line {num_ligne}: {nom}")
				if nom not in ("ID", "Diagnosis"):
					try:
						cellule = float(cellule)
					except ValueError:
						raise ValueError(f"Invalid number at line {num_ligne}: {nom} ('{cellule}')")
				colonnes[nom].append(cellule)
	if not any(colonnes.values()):
		raise ValueError("CSV file is empty")
	return colonnes


