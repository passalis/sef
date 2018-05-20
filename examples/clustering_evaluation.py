from sklearn.decomposition import PCA
import numpy as np
from sef_dr import LinearSEF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, calinski_harabaz_score
from sklearn.cluster import KMeans
from sef_dr.datasets import dataset_loader


def evaluate_clustering_solution(labels, predictions):
    """
    Calculates several clustering metrics
    :param data:
    :param labels:
    :param predictions:
    :return:
    """
    arand = adjusted_rand_score(labels, predictions)
    nmi = normalized_mutual_info_score(labels, predictions)
    homogeneity = homogeneity_score(labels, predictions)
    completness = completeness_score(labels, predictions)
    fowlkes = fowlkes_mallows_score(labels, predictions)

    metrics = [arand, nmi, homogeneity, completness, fowlkes]
    print(metrics)
    metrics = np.asarray(metrics)
    return metrics


def evaluate_methods(x_train, y_train, seed=1, n_iters=50, regularizer=0, in_class=0.9, out_of_class=0.1):
    # Avoid accidental zero seed
    np.random.seed(seed + 1)

    Nc = len(np.unique(y_train))
    n_clusters = np.int(Nc)
    n_dim = np.int(50)
    n_discriminative_clusters = np.int(1.2 * n_clusters)

    # Evaluate baseline k-means
    print("k-means")
    model = KMeans(n_clusters=n_clusters, n_init=3)
    model.fit(x_train)
    evaluate_clustering_solution(y_train, model.predict(x_train))

    # PCA
    pca = PCA(n_components=n_dim)
    x_train_pca = pca.fit_transform(x_train)

    print("k-means (pca)")
    model = KMeans(n_clusters=n_clusters, n_init=3)
    model.fit(x_train_pca)
    evaluate_clustering_solution(y_train, model.predict(x_train_pca))

    # Generate the supervised targets
    model = KMeans(n_clusters=n_discriminative_clusters, n_jobs=4, n_init=3)
    model.fit(x_train)
    targets = model.predict(x_train)

    # LDA
    lda = LinearDiscriminantAnalysis(n_components=min(Nc - 1, n_dim))
    x_train_lda = lda.fit_transform(x_train, targets)

    print("k-means (lda)")
    model = KMeans(n_clusters=n_clusters, n_init=3)
    model.fit(x_train_lda)
    evaluate_clustering_solution(y_train, model.predict(x_train_lda))

    # Proposed
    sef = LinearSEF(x_train.shape[1], output_dimensionality=n_dim, scaler=None)
    sef.cuda()
    target_params = {'in_class_similarity': in_class, 'bewteen_class_similarity': out_of_class}
    # Initialize sef
    sef.fit(data=x_train[:1000], target_labels=targets[:1000], target='supervised', epochs=1,
            batch_size=128, learning_rate=0.001, regularizer_weight=regularizer, target_params=target_params)
    # Run the optimization
    loss = sef.fit(data=x_train, target_labels=targets, target='supervised', epochs=n_iters,
                   batch_size=128, verbose=True, learning_rate=0.001, regularizer_weight=regularizer,
                   target_params=target_params,
                   warm_start=True)
    x_train_sef = sef.transform(x_train)

    print("k-means (sef)")
    model = KMeans(n_clusters=n_clusters, n_init=3)
    model.fit(x_train_sef)
    evaluate_clustering_solution(y_train, model.predict(x_train_sef))


def run_evaluation_on_dataset(dataset='mnist', seed=1):
    reg = 1
    in_class = 0.8
    out_of_class = 0.2

    if dataset == 'yale':
        reg = 0.00001
        in_class = 0.5
        out_of_class = 0.3

    x_train, y_train, x_test, y_test = dataset_loader('./data', dataset, seed=seed, )
    evaluate_methods(x_train, y_train, seed=seed, regularizer=reg, in_class=in_class, out_of_class=out_of_class)


if __name__ == '__main__':
    run_evaluation_on_dataset('15scene')
    run_evaluation_on_dataset('corel')
    run_evaluation_on_dataset('yale')
