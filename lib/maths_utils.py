# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    maths_utils.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gajanvie <gajanvie@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/10/08 14:39:52 by gajanvie          #+#    #+#              #
#    Updated: 2026/10/23 10:18:30 by gajanvie         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

def ft_count(valeurs):
	n = 0
	for _ in valeurs:
		n += 1
	return n


def ft_sum(valeurs):
	total = 0.0
	for v in valeurs:
		total += v
	return total

def ft_mean(valeurs):
	n = ft_count(valeurs)
	if n == 0:
		raise ValueError("ft_mean : liste vide")
	return ft_sum(valeurs) / n

def ft_std(valeurs):
	n = ft_count(valeurs)
	if n < 2:
		raise ValueError("ft_std : faut au moins 2 valeurs")
	mu = ft_mean(valeurs)
	somme = 0.0
	for v in valeurs:
		somme += (v - mu) ** 2
	return (somme / (n - 1)) ** 0.5


def ft_min(valeurs):
	if ft_count(valeurs) == 0:
		raise ValueError("ft_min : liste vide")
	m = valeurs[0]
	for v in valeurs:
		if v < m:
			m = v
	return m


def ft_max(valeurs):
	if ft_count(valeurs) == 0:
		raise ValueError("ft_max : liste vide")
	m = valeurs[0]
	for v in valeurs:
		if v > m:
			m = v
	return m

def ft_sort(valeurs):
	r = list(valeurs)
	for i in range(1, ft_count(r)):
		cle = r[i]
		j = i - 1
		while j >= 0 and r[j] > cle:
			r[j + 1] = r[j]
			j -= 1
		r[j + 1] = cle
	return r



def ft_variance(valeurs):
	"""variance echantllon (n - 1)"""
	return ft_std(valeurs) ** 2

