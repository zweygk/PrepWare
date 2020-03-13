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

    def compute_optimal_meal(self, minimumIntakeObj, nItems=2):
        print('Bzzzz. Computing optimal meal using Lasso objective.')
        
        df_ = self.df[list(asdict(self.mappings).values())]
        df_ = df_.transpose().dropna(axis=1, thresh=4)
        
        meal_combos = self.get_meal_combinations(df_, nItems)
        y = list(asdict(minimumIntakeObj).values())
        
        score = 0
        coef = None
        best_combo = None
        
        for combo in meal_combos:
            X = df_[list(combo)]
            reg = Lasso(positive=True, alpha=0.001).fit(X, y)
            goodness = reg.score(X, y)
            if goodness > score and self.any_gramcoef_bigger_than_remaining_weight(combo, reg.coef_) == False:
                score = goodness
                coef = reg.coef_
                best_combo = combo
        
        weights_as_grams = [ 100*w for w in coef ]

        print('Done.\n\nOptimal meal combination of {n} items is \n'.format(n = nItems))
        print(best_combo)
        print('With weights {}'.format(coef))
        print('or grams: {}'.format(weights_as_grams))
        print('Resulting in a R² of {} for fitting the minimum daily intake.'. format(score))
        
        return best_combo, coef, score

