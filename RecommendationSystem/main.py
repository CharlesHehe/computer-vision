# @title Imports (run this cell)
from __future__ import print_function

import numpy as np
import pandas as pd
import collections
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow._api.v2.compat.v1 as tf
import altair as alt
from urllib.request import urlretrieve
import zipfile
from datetime import datetime

tf.compat.v1.disable_eager_execution()
USER_RATINGS = False
# Add some convenience functions to Pandas DataFrame.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format


def mask(df, key, function):
    """Returns a filtered dataframe, by applying function to key"""
    return df[function(df[key])]


def flatten_cols(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df


pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols


# @title CFModel helper class (run this cell)
class CFModel(object):
    """Simple class that represents a collaborative filtering model"""

    def __init__(self, embedding_vars, loss, metrics=None):
        """Initializes a CFModel.
        Args:
          embedding_vars: A dictionary of tf.Variables.
          loss: A float Tensor. The loss to optimize.
          metrics: optional list of dictionaries of Tensors. The metrics in each
            dictionary will be plotted in a separate figure during training.
        """
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k: None for k in embedding_vars}
        self._session = None

    @property
    def embeddings(self):
        """The embeddings dictionary."""
        return self._embeddings

    def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
              optimizer=tf.train.GradientDescentOptimizer):
        """Trains the model.
        Args:
          iterations: number of iterations to run.
          learning_rate: optimizer learning rate.
          plot_results: whether to plot the results at the end of training.
          optimizer: the optimizer to use. Default to GradientDescentOptimizer.
        Returns:
          The metrics dictionary evaluated at the last iteration.
        """
        with self._loss.graph.as_default():
            opt = optimizer(learning_rate)
            train_op = opt.minimize(self._loss)
            local_init_op = tf.group(
                tf.variables_initializer(opt.variables()),
                tf.local_variables_initializer())
            if self._session is None:
                self._session = tf.Session()
                with self._session.as_default():
                    self._session.run(tf.global_variables_initializer())
                    self._session.run(tf.tables_initializer())
                    tf.train.start_queue_runners()

        with self._session.as_default():
            local_init_op.run()
            iterations = []
            metrics = self._metrics or ({},)
            metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

            # Train and append results.
            for i in range(num_iterations + 1):
                _, results = self._session.run((train_op, metrics))
                if (i % 10 == 0) or i == num_iterations:
                    print("\r iteration %d: " % i + ", ".join(
                        ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                          end='')
                    iterations.append(i)
                    for metric_val, result in zip(metrics_vals, results):
                        for k, v in result.items():
                            metric_val[k].append(v)

            for k, v in self._embedding_vars.items():
                self._embeddings[k] = v.eval()

            if plot_results:
                # Plot the metrics.
                num_subplots = len(metrics) + 1
                fig = plt.figure()
                fig.set_size_inches(num_subplots * 10, 8)
                for i, metric_vals in enumerate(metrics_vals):
                    ax = fig.add_subplot(1, num_subplots, i + 1)
                    for k, v in metric_vals.items():
                        ax.plot(iterations, v, label=k)
                    ax.set_xlim([1, num_iterations])
                    ax.legend()
            return results


# Load each data set (users, movies, and ratings).
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(
    'ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# The movies file contains a binary feature for each genre.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = [
                  'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
              ] + genre_cols
movies = pd.read_csv(
    'ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')

# Since the ids start at 1, we shift them to start at 0.
users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

# Compute the number of movies to which a genre is assigned.
genre_occurences = movies[genre_cols].sum().to_dict()

# Compute the number of movies to which a genre is assigned.
genre_occurences = movies[genre_cols].sum().to_dict()


# Since some movies can belong to more than one genre, we create different
# 'genre' columns as follows:
# - all_genres: all the active genres of the movie.
# - genre: randomly sampled from the active genres.
def mark_genres(movies, genres):
    def get_random_genre(gs):
        active = [genre for genre, g in zip(genres, gs) if g == 1]
        if len(active) == 0:
            return 'Other'
        return np.random.choice(active)

    def get_all_genres(gs):
        active = [genre for genre, g in zip(genres, gs) if g == 1]
        if len(active) == 0:
            return 'Other'
        return '-'.join(active)

    for genre in genres:
        movies[genre]*movies[genre]
    movies['genre'] = [
        get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
    movies['all_genres'] = [
        get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]


mark_genres(movies, genre_cols)

# Create one merged DataFrame containing all the movielens data.
movielens = ratings.merge(movies, on='movie_id').merge(users, on='user_id')

indices = ratings[['user_id', 'movie_id']].values
values = ratings['rating'].values

movies_ratings = movies.merge(
    ratings
        .groupby('movie_id', as_index=False)
        .agg({'rating': ['count', 'mean']})
        .flatten_cols(),
    on='movie_id')

genre_filter = alt.selection_multi(fields=['genre'])
genre_chart = alt.Chart().mark_bar().encode(
    x="count()",
    y=alt.Y('genre'),
    color=alt.condition(
        genre_filter,
        alt.Color("genre:N"),
        alt.value('lightgray'))
).properties(height=300, selection=genre_filter)

(movies_ratings[['title', 'rating count', 'rating mean']]
 .sort_values('rating count', ascending=False)
 .head(10))

(movies_ratings[['title', 'rating count', 'rating mean']]
 .mask('rating count', lambda x: x > 20)
 .sort_values('rating mean', ascending=False)
 .head(10))


# Utility to split the data into training and test sets.
def split_dataframe(df, holdout_fraction=0.1):
    """Splits a DataFrame into training and test sets.
    Args:
      df: a dataframe.
      holdout_fraction: fraction of dataframe rows to use in the test set.
    Returns:
      train: dataframe for training
      test: dataframe for testing
    """
    test = df.sample(frac=holdout_fraction, replace=False)
    train = df[~df.index.isin(test.index)]
    return train, test


# @title Solution
def build_rating_sparse_tensor(ratings_df):
    """
    Args:
      ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
    Returns:
      a tf.SparseTensor representing the ratings matrix.
    """
    indices = ratings_df[['user_id', 'movie_id']].values
    values = ratings_df['rating'].values
    return tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[users.shape[0], movies.shape[0]])


# @title Solution
def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
    """
    Args:
      sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
      user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
        dimension, such that U_i is the embedding of user i.
      movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
        dimension, such that V_j is the embedding of movie j.
    Returns:
      A scalar Tensor representing the MSE between the true ratings and the
        model's predictions.
    """
    predictions = tf.gather_nd(
        tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
        sparse_ratings.indices)
    loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
    return loss


# @title Solution
def build_model(ratings, embedding_dim=3, init_stddev=1.):
    """
    Args:
      ratings: a DataFrame of the ratings
      embedding_dim: the dimension of the embedding vectors.
      init_stddev: float, the standard deviation of the random initial embeddings.
    Returns:
      model: a CFModel.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings)
    A_test = build_rating_sparse_tensor(test_ratings)
    # Initialize the embeddings using a normal distribution.
    U = tf.Variable(tf.random_normal(
        [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
    train_loss = sparse_mean_square_error(A_train, U, V)
    test_loss = sparse_mean_square_error(A_test, U, V)
    metrics = {
        'train_error': train_loss,
        'test_error': test_loss
    }
    embeddings = {
        "user_id": U,
        "movie_id": V
    }

    # run the graph
    return CFModel(embeddings, train_loss, [metrics])


# Build the CF model and train it.
model = build_model(ratings, embedding_dim=30, init_stddev=0.5)
model.train(num_iterations=1000, learning_rate=10.)

# @title Solution
DOT = 'dot'
COSINE = 'cosine'


def compute_scores(query_embedding, item_embeddings, measure=DOT):
    """Computes the scores of the candidates given a query.
    Args:
      query_embedding: a vector of shape [k], representing the query embedding.
      item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
      measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
      scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == COSINE:
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores


# @title User recommendations and nearest neighbors (run this cell)
def user_recommendations(model, measure=DOT, exclude_rated=False, k=6):
    if True:
        scores = compute_scores(
            model.embeddings["user_id"][942], model.embeddings["movie_id"], measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'movie_id': movies['movie_id'],
            'titles': movies['title'],
            'genres': movies['all_genres'],
        })
        if exclude_rated:
            # remove movies that are already rated
            rated_movies = ratings[ratings.user_id == "943"]["movie_id"].values
            df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]
        display.display(df.sort_values([score_key], ascending=False).head(k))


def movie_neighbors(model, title_substring, measure=DOT, k=6):
    # Search for movie ids that match the given substring.
    ids = movies[movies['title'].str.contains(title_substring)].index.values
    titles = movies.iloc[ids]['title'].values
    if len(titles) == 0:
        raise ValueError("Found no movies with title %s" % title_substring)
    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching movie. Other candidates: {}]".format(
            ", ".join(titles[1:])))
    movie_id = ids[0]
    scores = compute_scores(
        model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
        measure)
    score_key = measure + ' score'
    df = pd.DataFrame({
        score_key: list(scores),
        'titles': movies['title'],
        'genres': movies['all_genres']
    })
    display.display(df.sort_values([score_key], ascending=False).head(k))


user_recommendations(model, measure=COSINE, k=5)

movie_neighbors(model, "Aladdin", DOT)
movie_neighbors(model, "Aladdin", COSINE)


# @title Embedding Visualization code (run this cell)

def movie_embedding_norm(models):
    """Visualizes the norm and number of ratings of the movie embeddings.
    Args:
      model: A MFModel object.
    """
    if not isinstance(models, list):
        models = [models]
    df = pd.DataFrame({
        'title': movies['title'],
        'genre': movies['genre'],
        'num_ratings': movies_ratings['rating count'],
    })
    charts = []
    brush = alt.selection_interval()
    for i, model in enumerate(models):
        norm_key = 'norm' + str(i)
        df[norm_key] = np.linalg.norm(model.embeddings["movie_id"], axis=1)
        nearest = alt.selection(
            type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
            empty='none')
        base = alt.Chart().mark_circle().encode(
            x='num_ratings',
            y=norm_key,
            color=alt.condition(brush, alt.value('#4c78a8'), alt.value('lightgray'))
        ).properties(
            selection=nearest).add_selection(brush)
        text = alt.Chart().mark_text(align='center', dx=5, dy=-5).encode(
            x='num_ratings', y=norm_key,
            text=alt.condition(nearest, 'title', alt.value('')))
        charts.append(alt.layer(base, text))
    return alt.hconcat(*charts, data=df)


def visualize_movie_embeddings(data, x, y):
    nearest = alt.selection(
        type='single', encodings=['x', 'y'], on='mouseover', nearest=True,
        empty='none')
    base = alt.Chart().mark_circle().encode(
        x=x,
        y=y,
        color=alt.condition(genre_filter, "genre", alt.value("whitesmoke")),
    ).properties(
        width=600,
        height=600,
        selection=nearest)
    text = alt.Chart().mark_text(align='left', dx=5, dy=-5).encode(
        x=x,
        y=y,
        text=alt.condition(nearest, 'title', alt.value('')))
    return alt.hconcat(alt.layer(base, text), genre_chart, data=data)


def tsne_movie_embeddings(model):
    """Visualizes the movie embeddings, projected using t-SNE with Cosine measure.
    Args:
      model: A MFModel object.
    """
    tsne = sklearn.manifold.TSNE(
        n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
        init='pca', verbose=True, n_iter=400)

    print('Running t-SNE...')
    V_proj = tsne.fit_transform(model.embeddings["movie_id"])
    movies.loc[:, 'x'] = V_proj[:, 0]
    movies.loc[:, 'y'] = V_proj[:, 1]
    return visualize_movie_embeddings(movies, 'x', 'y')


movie_embedding_norm(model)

# @title Solution
model_lowinit = build_model(ratings, embedding_dim=30, init_stddev=0.05)
model_lowinit.train(num_iterations=1000, learning_rate=10.)
movie_neighbors(model_lowinit, "Aladdin", DOT)
movie_neighbors(model_lowinit, "Aladdin", COSINE)
movie_embedding_norm([model, model_lowinit])

tsne_movie_embeddings(model_lowinit)


# @title Solution
def gravity(U, V):
    """Creates a gravity loss given two embedding matrices."""
    return 1. / (U.shape[0] * V.shape[0]) * tf.reduce_sum(
        tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))


def build_regularized_model(
        ratings, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
        init_stddev=0.1):
    """
    Args:
      ratings: the DataFrame of movie ratings.
      embedding_dim: The dimension of the embedding space.
      regularization_coeff: The regularization coefficient lambda.
      gravity_coeff: The gravity regularization coefficient lambda_g.
    Returns:
      A CFModel object that uses a regularized loss.
    """
    # Split the ratings DataFrame into train and test.
    train_ratings, test_ratings = split_dataframe(ratings)
    # SparseTensor representation of the train and test datasets.
    A_train = build_rating_sparse_tensor(train_ratings)
    A_test = build_rating_sparse_tensor(test_ratings)
    U = tf.Variable(tf.random_normal(
        [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
    V = tf.Variable(tf.random_normal(
        [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

    error_train = sparse_mean_square_error(A_train, U, V)
    error_test = sparse_mean_square_error(A_test, U, V)
    gravity_loss = gravity_coeff * gravity(U, V)
    regularization_loss = regularization_coeff * (
            tf.reduce_sum(U * U) / U.shape[0] + tf.reduce_sum(V * V) / V.shape[0])
    total_loss = error_train + regularization_loss + gravity_loss
    losses = {
        'train_error_observed': error_train,
        'test_error_observed': error_test,
    }
    loss_components = {
        'observed_loss': error_train,
        'regularization_loss': regularization_loss,
        'gravity_loss': gravity_loss,
    }
    embeddings = {"user_id": U, "movie_id": V}

    return CFModel(embeddings, total_loss, [losses, loss_components])


reg_model = build_regularized_model(
    ratings, regularization_coeff=0.1, gravity_coeff=1.0, embedding_dim=35,
    init_stddev=.05)
reg_model.train(num_iterations=2000, learning_rate=20.)

user_recommendations(reg_model, DOT, exclude_rated=True, k=10)
movie_neighbors(reg_model, "Aladdin", DOT)
movie_neighbors(reg_model, "Aladdin", COSINE)
movie_embedding_norm([model, model_lowinit, reg_model])
# Visualize the embeddings
tsne_movie_embeddings(reg_model)

rated_movies = (ratings[["user_id", "movie_id"]]
                .groupby("user_id", as_index=False)
                .aggregate(lambda x: list(x)))
rated_movies.head()

# @title Batch generation code (run this cell)
years_dict = {
    movie: year for movie, year in zip(movies["movie_id"], movies["year"])
}
genres_dict = {
    movie: genres.split('-')
    for movie, genres in zip(movies["movie_id"], movies["all_genres"])
}


def make_batch(ratings, batch_size):
    """Creates a batch of examples.
    Args:
      ratings: A DataFrame of ratings such that examples["movie_id"] is a list of
        movies rated by a user.
      batch_size: The batch size.
    """

    def pad(x, fill):
        return pd.DataFrame.from_dict(x).fillna(fill).values

    movie = []
    year = []
    genre = []
    label = []
    for movie_ids in ratings["movie_id"].values:
        movie.append(movie_ids)
        genre.append([x for movie_id in movie_ids for x in genres_dict[movie_id]])
        year.append([years_dict[movie_id] for movie_id in movie_ids])
        label.append([int(movie_id) for movie_id in movie_ids])
    features = {
        "movie_id": pad(movie, ""),
        "year": pad(year, ""),
        "genre": pad(genre, ""),
        "label": pad(label, -1)
    }
    batch = (
        tf.data.Dataset.from_tensor_slices(features)
            .shuffle(1000)
            .repeat()
            .batch(batch_size)
            .make_one_shot_iterator()
            .get_next())
    return batch


def select_random(x):
    """Selectes a random elements from each row of x."""

    def to_float(x):
        return tf.cast(x, tf.float32)

    def to_int(x):
        return tf.cast(x, tf.int64)

    batch_size = tf.shape(x)[0]
    rn = tf.range(batch_size)
    nnz = to_float(tf.count_nonzero(x >= 0, axis=1))
    rnd = tf.random_uniform([batch_size])
    ids = tf.stack([to_int(rn), to_int(nnz * rnd)], axis=1)
    return to_int(tf.gather_nd(x, ids))


# @title Solution
def softmax_loss(user_embeddings, movie_embeddings, labels):
    """Returns the cross-entropy loss of the softmax model.
    Args:
      user_embeddings: A tensor of shape [batch_size, embedding_dim].
      movie_embeddings: A tensor of shape [num_movies, embedding_dim].
      labels: A tensor of [batch_size], such that labels[i] is the target label
        for example i.
    Returns:
      The mean cross-entropy loss.
    """
    # Verify that the embddings have compatible dimensions
    user_emb_dim = user_embeddings.shape[1]
    movie_emb_dim = movie_embeddings.shape[1]
    if user_emb_dim != movie_emb_dim:
        raise ValueError(
            "The user embedding dimension %d should match the movie embedding "
            "dimension % d" % (user_emb_dim, movie_emb_dim))

    logits = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    return loss


# @title Solution

def build_softmax_model(rated_movies, embedding_cols, hidden_dims):
    """Builds a Softmax model for MovieLens.
    Args:
      rated_movies: DataFrame of traing examples.
      embedding_cols: A dictionary mapping feature names (string) to embedding
        column objects. This will be used in tf.feature_column.input_layer() to
        create the input layer.
      hidden_dims: int list of the dimensions of the hidden layers.
    Returns:
      A CFModel object.
    """

    def create_network(features):
        """Maps input features dictionary to user embeddings.
        Args:
          features: A dictionary of input string tensors.
        Returns:
          outputs: A tensor of shape [batch_size, embedding_dim].
        """
        # Create a bag-of-words embedding for each sparse feature.
        inputs = tf.feature_column.input_layer(features, embedding_cols)
        # Hidden layers.
        input_dim = inputs.shape[1]
        for i, output_dim in enumerate(hidden_dims):
            w = tf.get_variable(
                "hidden%d_w_" % i, shape=[input_dim, output_dim],
                initializer=tf.truncated_normal_initializer(
                    stddev=1. / np.sqrt(output_dim))) / 10.
            outputs = tf.matmul(inputs, w)
            input_dim = output_dim
            inputs = outputs
        return outputs

    train_rated_movies, test_rated_movies = split_dataframe(rated_movies)
    train_batch = make_batch(train_rated_movies, 200)
    test_batch = make_batch(test_rated_movies, 100)

    with tf.variable_scope("model", reuse=False):
        # Train
        train_user_embeddings = create_network(train_batch)
        train_labels = select_random(train_batch["label"])
    with tf.variable_scope("model", reuse=True):
        # Test
        test_user_embeddings = create_network(test_batch)
        test_labels = select_random(test_batch["label"])
        movie_embeddings = tf.get_variable(
            "input_layer/movie_id_embedding/embedding_weights")

    test_loss = softmax_loss(
        test_user_embeddings, movie_embeddings, test_labels)
    train_loss = softmax_loss(
        train_user_embeddings, movie_embeddings, train_labels)
    _, test_precision_at_10 = tf.metrics.precision_at_k(
        labels=test_labels,
        predictions=tf.matmul(test_user_embeddings, movie_embeddings, transpose_b=True),
        k=10)

    metrics = (
        {"train_loss": train_loss, "test_loss": test_loss},
        {"test_precision_at_10": test_precision_at_10}
    )
    embeddings = {"movie_id": movie_embeddings}
    return CFModel(embeddings, train_loss, metrics)


# Create feature embedding columns
def make_embedding_col(key, embedding_dim):
    categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=key, vocabulary_list=list(set(movies[key].values)), num_oov_buckets=0)
    return tf.feature_column.embedding_column(
        categorical_column=categorical_col, dimension=embedding_dim,
        # default initializer: trancated normal with stddev=1/sqrt(dimension)
        combiner='mean')


with tf.Graph().as_default():
    softmax_model = build_softmax_model(
        rated_movies,
        embedding_cols=[
            make_embedding_col("movie_id", 35),
            make_embedding_col("genre", 3),
            make_embedding_col("year", 2),
        ],
        hidden_dims=[35])

softmax_model.train(
    learning_rate=8., num_iterations=3000, optimizer=tf.train.AdagradOptimizer)


movie_neighbors(softmax_model, "Aladdin", DOT)
movie_neighbors(softmax_model, "Aladdin", COSINE)
movie_embedding_norm([reg_model, softmax_model])
tsne_movie_embeddings(softmax_model)
