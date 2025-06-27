import numpy as np

class GaussianNaiveBayes:
  def fit(self, X, y):
    self.classes = np.unique(y)
    n_features = X.shape[1]

    self.means = {}
    self.vars = {}
    self.priors = {}

    for c in self.classes:
      X_c = X[y == c]
      self.means[c] = X_c.mean(axis=0)
      self.vars[c] = X_c.var(axis=0)
      self.priors[c] = X_c.shape[0] / X.shape[0]

  def _gaussian_pdf(self, x, mean, var):
    eps = 1e-9  
    coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
    exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
    return coeff * exponent

  def predict(self, X):

    y_pred = []

    for x in X:
      posteriors = []

      for c in self.classes:
        prior = np.log(self.priors[c]) 
        likelihoods = self._gaussian_pdf(x, self.means[c], self.vars[c])
        total_likelihood = np.sum(np.log(likelihoods + 1e-9))
        posterior = prior + total_likelihood
        posteriors.append(posterior)

      y_pred.append(self.classes[np.argmax(posteriors)])

    return np.array(y_pred)


#all we do is save std and mean, and priors
#rest is the same as categorical, just log sums instead of probability multiplication.