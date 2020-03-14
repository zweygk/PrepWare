import pandas as pd
import numpy as np
import itertools

from dataclasses import dataclass, asdict
from sklearn.linear_model import Lasso

from DailyMinimumIntake import DailyMinimumIntake

@dataclass
class NourishmentColumnMappings:
    """Maps nourishment type to dataframe column."""
    
    calories : str = ''
    protein : str = ''
    sugar : str = ''
    carbohydrates : str = ''
    fat : str = ''
        
    def __post_init__(self):
        self.calories = 'kcal_per_100g'
        self.protein = 'grams_protein_per_100g'
        self.sugar = 'grams_sugar_per_100g'
        self.carbohydrates = 'grams_carbohydrates_per_100g'
        self.fat = 'grams_fat_per_100g'
                

class SurvivalBot:
    """SurvivalBot is your best friend when SHTF"""
    
    def __init__(self, csvpath):
        self.df = pd.read_csv(csvpath, sep=';').set_index('item')
        self.mappings = NourishmentColumnMappings()
            
    def get_sum(self, nourishment, consumed = False):
        if not consumed:
            df_ = self.df[self.df['consumed'] == False]
        factors = df_['item_count']*df_['weight_grams']/100
        return (df_[asdict(self.mappings)[nourishment]] * factors).sum()
    
    def compute_days_left_one(self, nourishment_intake_tuple, consumed = False):
        left = self.get_sum(nourishment_intake_tuple[0], consumed)
        days_left = np.round(left/nourishment_intake_tuple[1], 1)
        return days_left
    
    def compute_days_left_all(self, minimumIntakeObj):
        return [ self.compute_days_left_one(x) for x in asdict(minimumIntakeObj).items() ]
            
    def get_days_left(self, minimumIntakeObj):
        print("Beep boop\n")
        days_left = self.compute_days_left_all(minimumIntakeObj)
        print('You have\n')
        for i, nourishment_type in enumerate(asdict(minimumIntakeObj).keys()):
            print('    - {d} days of {n}'.format(d=days_left[i], n=nourishment_type))
        print('\nleft.')
        
    def get_meal_combinations(self, df, r):
        return list(itertools.combinations(df.columns.values, r))
        
    def any_gramcoef_bigger_than_remaining_weight(self, items, coefs):
        l = []
        for i, item in enumerate(items):
            l.append((self.df.loc[item]['weight_grams'] * self.df.loc[item]['item_count']) < coefs[i]*100)
        return (True in l)


    def n_best_meals(self, n_meals, minimumIntakeObj, nItems=2, itemConstraints = None):
        print("Bzzz...Computing {nM} most optimal meal combos of {nI} items.\n".format(
            nM = n_meals, nI = nItems
        ))
        
        df_ = self.df[list(asdict(self.mappings).values())]
        df_ = df_.transpose().dropna(axis=1, thresh=4)
        
        meal_combos = self.get_meal_combinations(df_, nItems)
        y = list(asdict(minimumIntakeObj).values())

        scores, coefs, combos = [],[],[]
        len_meal_combos = len(meal_combos)

        for k, combo in enumerate(meal_combos):
            X = df_[list(combo)]
            reg = Lasso(positive=True, alpha=0.001).fit(X, y)
            coef = reg.coef_
            if self.any_gramcoef_bigger_than_remaining_weight(combo, coef) == False:
                if itemConstraints:
                    if set(itemConstraints).union(set(combo)) == set(combo):
                        scores.append(reg.score(X, y))
                        coefs.append(coef)
                        combos.append(combo)
                else:
                    scores.append(reg.score(X, y))
                    coefs.append(coef)
                    combos.append(combo)

            if (k+1) % 110 == 0: print("{}%".format(np.round((k+1)*100/len_meal_combos, 1)))

        indices = list(range(len(scores)))
        indices.sort(key=scores.__getitem__, reverse=True)

        scores_sorted = list(map(scores.__getitem__, indices))
        coefs_sorted = list(map(coefs.__getitem__, indices))
        combos_sorted = list(map(combos.__getitem__, indices))

        print('Done.\n\n')

        for i in range(n_meals):
            weight_as_grams = 100*coefs_sorted[i]
            print("--------- Meal {} ----------\n".format(i))
            print('Items: {}'.format(combos_sorted[i]))
            print("Grams: {}".format(weight_as_grams))
            print("R²: {}\n\n".format(scores_sorted[i]))

        return combos_sorted, coefs_sorted, scores_sorted