import copy
import random

from .pool_stats import level_percentage, chosen_stats, cost_star_values, base_pool_values
from .pool_stats import COST_1 as _COST_1_TEMPLATE
from .pool_stats import COST_2 as _COST_2_TEMPLATE
from .pool_stats import COST_3 as _COST_3_TEMPLATE
from .pool_stats import COST_4 as _COST_4_TEMPLATE
from .pool_stats import COST_5 as _COST_5_TEMPLATE
from .origin_class_stats import origin_class, chosen_exclude
from .config import LOGMESSAGES
from . import stats as stats


class pool:
	def __init__(self):
		self._costs = [
			copy.deepcopy(_COST_1_TEMPLATE),
			copy.deepcopy(_COST_2_TEMPLATE),
			copy.deepcopy(_COST_3_TEMPLATE),
			copy.deepcopy(_COST_4_TEMPLATE),
			copy.deepcopy(_COST_5_TEMPLATE),
		]
		self.num_cost_1 = 0
		self.num_cost_2 = 0
		self.num_cost_3 = 0
		self.num_cost_4 = 0
		self.num_cost_5 = 0
		self.reset()
		self.update_stats(allV=True)

	def _cost(self, idx):
		return self._costs[idx]

	def log_to_file_pool(self):
		if LOGMESSAGES:
			with open('log.txt', "a") as out:
				for i, label in enumerate(["COST 1", "COST 2", "COST 3", "COST 4", "COST 5"]):
					out.write(label + " \n")
					for key, value in self._costs[i].items():
						out.write('%s:%s\n' % (key, value))

	def pick_chosen(self, champion_name):
		chosen_type = random.choice(origin_class[champion_name])
		counter = 0
		while chosen_type in chosen_exclude:
			counter += 1
			if counter >= 50:
				chosen_type = None
				print("I should never be here with champion {}".format(champion_name))
				break
			chosen_type = random.choice(origin_class[champion_name])
		return chosen_type

	def reset(self):
		for i, template in enumerate([_COST_1_TEMPLATE, _COST_2_TEMPLATE, _COST_3_TEMPLATE, _COST_4_TEMPLATE, _COST_5_TEMPLATE]):
			self._costs[i].clear()
			self._costs[i].update(template)

	def return_hero(self, player):
		for i in range(len(player.board)):
			for k in range(len(player.board[0])):
				if player.board[i][k]:
					self.update_pool(player.board[i][k], 1)
		for i in range(len(player.bench)):
			if player.bench[i]:
				self.update_pool(player.bench[i], 1)

	def sample(self, player, num, idx=-1):
		if player is None:
			return [" " for _ in range(num)]
		ranInt = [0 for _ in range(num)]
		championOptions = [None for _ in range(num)]
		chosen = player.chosen
		chosen_index = -1
		if not chosen:
			if random.random() < .5:
				chosen_index = random.randint(0, 4)
		index = idx
		for i in range(0, num):
			if chosen_index != i:
				percents = level_percentage[player.level]
			else:
				percents = chosen_stats[player.level]
			valid_pick = False
			attempts = 0
			
			while not valid_pick and attempts < 100:
				ranInt[i] = random.random()
				
				if idx == -1:
					index = 0
					while ranInt[i] > percents[index]:
						index += 1
						if index > 4:
							print("ERROR with ranInt[i] = " + str(ranInt[i]) + " and player level" + str(player.level))
							break
				else:
					index = idx
					
				if index == 0 and self.num_cost_1 > 0:
					valid_pick = True
				elif index == 1 and self.num_cost_2 > 0:
					valid_pick = True
				elif index == 2 and self.num_cost_3 > 0:
					valid_pick = True
				elif index == 3 and self.num_cost_4 > 0:
					valid_pick = True
				elif index == 4 and self.num_cost_5 > 0:
					valid_pick = True
					
				if idx != -1 and not valid_pick:
					break
					
				attempts += 1
				
			if not valid_pick:
				championOptions[i] = " "
				continue

			counter = 0
			counterIndex = 0

			cost_pool = self._costs[index]
			cost_values = list(cost_pool.values())
			total = sum(cost_values)
			ranPoolInt = random.randint(0, total - 1) if total > 0 else 0
			while counter < ranPoolInt and counterIndex < len(cost_values):
				counter += cost_values[counterIndex]
				counterIndex += 1
			keys_list = list(cost_pool)
			championOptions[i] = keys_list[counterIndex - 1] if counterIndex > 0 and counterIndex <= len(keys_list) else " "

			if chosen_index == i:
				if championOptions[i] != " ":
					chosen_type = self.pick_chosen(championOptions[i])
					championOptions[i] = str(championOptions[i]) + "_" + chosen_type + "_c"
			index = idx
		return championOptions

	def update_pool(self, u_champion, direction):
		cost = u_champion.cost
		quantity = 3 ** (u_champion.stars - 1) * direction
		if u_champion.stars != 1:
			cost = stats.COST[u_champion.name]
		idx = cost - 1
		if 0 <= idx < 5:
			pool_dict = self._costs[idx]
			bound = base_pool_values[idx]
			pool_dict[u_champion.name] = pool_dict.get(u_champion.name, 0) + quantity
			if pool_dict[u_champion.name] < 0:
				pool_dict[u_champion.name] = 0
			elif pool_dict[u_champion.name] > bound:
				pool_dict[u_champion.name] = bound
			if idx == 0:
				self.update_stats(one=True)
			elif idx == 1:
				self.update_stats(two=True)
			elif idx == 2:
				self.update_stats(three=True)
			elif idx == 3:
				self.update_stats(four=True)
			elif idx == 4:
				self.update_stats(five=True)

	def update_stats(self, allV=False, one=False, two=False, three=False, four=False, five=False):
		if allV or one:
			self.num_cost_1 = sum(self._costs[0].values())
		if allV or two:
			self.num_cost_2 = sum(self._costs[1].values())
		if allV or three:
			self.num_cost_3 = sum(self._costs[2].values())
		if allV or four:
			self.num_cost_4 = sum(self._costs[3].values())
		if allV or five:
			self.num_cost_5 = sum(self._costs[4].values())
