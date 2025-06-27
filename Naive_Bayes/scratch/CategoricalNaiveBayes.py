import pandas as pd

class SimpleNaiveBayes:
  def __init__(self):
    self.results = {}
    self.prior_yes = None
    self.prior_no = None

  def fit(self, data, target_col='play'):
    self.results = {}
    feature_cols = [col for col in data.columns if col != target_col]

    counts = data[target_col].value_counts(normalize=True)
    self.prior_yes = counts.get('Yes', 0)
    self.prior_no = counts.get('No', 0)

    for col in feature_cols:
      crosstab = pd.crosstab(data[col], data[target_col])
      total_no = crosstab['No'].sum()
      total_yes = crosstab['Yes'].sum()

      for val in crosstab.index:
        self.results[f'P{val}Yes'] = crosstab.loc[val, 'Yes'] / total_yes
        self.results[f'P{val}No'] = crosstab.loc[val, 'No'] / total_no

  def predict(self, input_data):
    p_yes = self.prior_yes
    p_no = self.prior_no

    for val in input_data:
      y_key = f'P{val}Yes'
      n_key = f'P{val}No'
      p_yes *= self.results.get(y_key, 1e-6)
      p_no *= self.results.get(n_key, 1e-6)

    return 'no' if p_no > p_yes else 'yes'
