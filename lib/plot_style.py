# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_style.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gajanvie <gajanvie@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/10/08 17:12:36 by gajanvie          #+#    #+#              #
#    Updated: 2026/10/08 17:14:17 by gajanvie         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt


FOND = "#0f1117"
FOND_AXES = "#171a23"
TEXTE = "#e6e6e6"
GRILLE = "#ffffff"
ACCENT = "#2ECC71"


def appliquer_style():
	plt.rcParams.update({
		"figure.facecolor": FOND,
		"savefig.facecolor": FOND,
		"axes.facecolor": FOND_AXES,
		"axes.edgecolor": "#3a3f4d",
		"axes.labelcolor": TEXTE,
		"axes.titlecolor": TEXTE,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.grid": True,
		"grid.color": GRILLE,
		"grid.alpha": 0.08,
		"text.color": TEXTE,
		"xtick.color": TEXTE,
		"ytick.color": TEXTE,
		"axes.titlesize": 10,
		"axes.titleweight": "bold",
		"font.size": 9,
		"legend.frameon": False,
		"legend.labelcolor": TEXTE,
	})