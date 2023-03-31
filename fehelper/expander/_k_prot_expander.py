import numpy as np
from kmodes.kprototypes import KPrototypes

RANDOM_SEED = 2022


class KProtExpander:
    """
    Transforms mixed data into k-prototypes cluster.

    This transformer runs k-prototypes on the input data and converts each data
    point into the ID of the closest cluster. If a target variable is present,
    it is scaled and included as input to k-prototypes in order to derive clusters
    that obey the classification boundary as well as group similar points together.
    """

    def __init__(self, k=2, target_scale=5.0,
                 categorical=None,
                 random_state=RANDOM_SEED):
        self.k = k
        self.target_scale = target_scale
        self.categorical = categorical
        self.random_state = random_state
        self.kprot_model = None
        self.cluster_centroids_ = None

    def fit(self, X, y=None):
        """Runs k-prototypes on the input data and finds centroids."""

        if y is None:
            # No target variable, just do plain k-prototypes
            kprot_model = KPrototypes(n_clusters=self.k,
                                      n_jobs=-1,
                                      random_state=self.random_state)
            kprot_model.fit(X, categorical=self.categorical)

            self.kprot_model = kprot_model
            self.cluster_centroids_ = kprot_model.cluster_centroids_
            return self

        # There is target information. Apply appropriate scaling and include
        # it in the input data to k-prototypes.
        data_with_target = np.hstack((X, y[:, np.newaxis] * self.target_scale))

        # Build a pre-training k-prototypes model on data and target
        kprot_model_pretrain = KPrototypes(n_clusters=self.k,
                                           n_jobs=-1,
                                           random_state=self.random_state)
        kprot_model_pretrain.fit(data_with_target, categorical=self.categorical)

        # Run k-prototypes a second time to get the clusters in the original space
        # without target info. Initialize using centroids found in pre-training.
        # Go through a single iteration of cluster assignment and centroid recomputation.
        kprot_model = KPrototypes(n_clusters=self.k,
                                  # For KPrototypes, we need to specify the cluster centroids for
                                  # numerical and categorical columns, notice that the numerical
                                  # part should exclude the target info
                                  init=[kprot_model_pretrain.cluster_centroids_[:,
                                        [i for i in range(kprot_model_pretrain.cluster_centroids_.shape[1] - 1)
                                         if i not in self.categorical]],
                                        kprot_model_pretrain.cluster_centroids_[:, self.categorical]],
                                  n_init=1,
                                  max_iter=1)
        kprot_model.fit(X, categorical=self.categorical)

        self.kprot_model = kprot_model
        self.cluster_centroids_ = kprot_model.cluster_centroids_
        return self

    def transform(self, X, y=None):
        """Outputs the closest cluster ID for each input data point.
        """
        clusters = self.kprot_model.predict(X, categorical=self.categorical)
        return clusters[:, np.newaxis]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
