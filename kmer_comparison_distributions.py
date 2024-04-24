import tensorflow as tf
from numpy import pi

pi = tf.cast(pi, tf.float64)


class ModelDistribution:
    # a distribution descriptor for (m,z) pair implemented IN TENSORFLOW
    # abstract model

    @staticmethod
    def calculate_probability(x, params):
        # params is a LIST of parameters (always a list)
        return None

    @staticmethod
    def estimate_initial_parameters(values, additional_info):
        return None

    @staticmethod
    def get_name():
        return "abstract"

    @staticmethod
    def validify_params(suggested_params):
        return suggested_params


class ExponentialDistribution(ModelDistribution):
    def __init__(self):
        super().__init__("exponential")

    @staticmethod
    def calculate_probability(x, params):
        x = tf.cast(x, tf.float64)
        # Pr[x_i] = Exp(x_i); Exp(x_i) = lambda exp(-lambda . x)
        lambda_param = params[0]
        probability = lambda_param * tf.math.exp(x * (- lambda_param))
        probability = tf.maximum(0, probability)
        probability = tf.minimum(1, probability)
        return probability

    @staticmethod
    def get_name():
        return "exponential"

    @staticmethod
    def estimate_initial_parameters(values, additional_info):
        subset_size = int(0.33 * len(values))
        subset = tf.random.shuffle(values)[:subset_size]
        lambda_param = subset_size / tf.reduce_sum(subset)
        return [tf.cast(lambda_param, tf.float64)]

    @staticmethod
    def validify_params(suggested_params):
        min_value = 1e-6
        return tf.maximum(suggested_params, min_value)


class UniformDistribution(ModelDistribution):
    @staticmethod
    def calculate_probability(x, params):
        max_n = params[0]
        probas = tf.ones_like(x, dtype=tf.float64) / max_n
        return probas

    @staticmethod
    def get_name():
        return "uniform"

    @staticmethod
    def estimate_initial_parameters(values, additional_info):
        return [tf.cast(tf.math.reduce_max(values) * 2,
                        tf.float64)]

    @staticmethod
    def validify_params(suggested_params):
        min_value = 1 + 1e-6
        par = tf.maximum(suggested_params, min_value)
        return par


class ReversedExponentialDistribution(ExponentialDistribution):
    @staticmethod
    def calculate_probability(x, params):
        probability = ExponentialDistribution.calculate_probability(x, params)
        return 1 - probability

    @staticmethod
    def estimate_initial_parameters(values, additional_info):
        subset_size = int(0.33 * len(values))
        subset = tf.random.shuffle(values)[:subset_size]
        lambda_param = subset_size / tf.reduce_sum(subset)
        return [tf.cast(lambda_param, tf.float64)]

    @staticmethod
    def get_name():
        return "reversed-exponential"

    @staticmethod
    def validify_params(suggested_params):
        return ExponentialDistribution.validify_params(suggested_params)

    @staticmethod
    def validify_params(suggested_params):
        min_value = 1e-6
        return tf.maximum(suggested_params, min_value)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class GaussianMixtureDistribution(ModelDistribution):
    @staticmethod
    def calculate_probability(x, params):
        x = tf.cast(x, tf.float64)
        # params: a list of triples, (weight, loc, scale)
        probability = tf.zeros_like(x)
        for i1, i2, i3 in chunks(list(range(params.shape[0])), 3):
            weight, loc, scale = params[i1], params[i2], params[i3]
            single_proba = tf.exp(-0.5 * tf.square((x - loc) / scale)) / (scale * tf.sqrt(2.0 * pi))
            probability = probability + (single_proba * weight)
        probability = tf.maximum(0, probability)
        probability = tf.minimum(1, probability)
        return probability

    @staticmethod
    def get_name():
        return "gaussian-mixture"

    @staticmethod
    def estimate_initial_parameters(values, additional_info):
        subset_size = int(0.33 * len(values))
        subset = tf.sort(tf.random.shuffle(values)[:subset_size])
        gmm_no = additional_info["gmm_models_no"]
        params = []
        for i in range(gmm_no):
            # weight, loc, scale
            params.append(tf.cast(1 / gmm_no, tf.float64))
            loc_index = tf.cast(tf.round((subset_size - 1) * (i / gmm_no)), tf.int32)
            loc = tf.cast(tf.gather(subset, loc_index), tf.float64)
            params.append(tf.cast(loc, tf.float64))
            params.append(tf.cast(5, tf.float64))  # any value
        return tf.stack(params)

    @staticmethod
    def validify_params(suggested_params):
        # gmm - weights are between 0 and 1 and sum to one, scale is strictly positive
        weights = []
        for i1, i2, i3 in chunks(list(range(suggested_params.shape[0])), 3):
            weight, loc, scale = suggested_params[i1], suggested_params[i2], suggested_params[i3]
            weights.append(tf.maximum(0, weight))
        Z = tf.reduce_sum(weights)

        params = []
        for i1, i2, i3 in chunks(list(range(suggested_params.shape[0])), 3):
            weight, loc, scale = suggested_params[i1], suggested_params[i2], suggested_params[i3]
            params.append(tf.maximum(0, weight / Z))
            params.append(loc)
            params.append(tf.maximum(1e-6, scale))

        return tf.stack(params)
