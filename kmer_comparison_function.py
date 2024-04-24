import numpy as np
import tensorflow as tf
from itertools import product
from collections import Counter


class KmerMetric:
    def __init__(self,
                 name, annot, metric_type,
                 probability_model
                 ):
        self.name = name
        self.annotation = annot
        self.metric_type = metric_type  # distance or similarity

        # probability_functions and stuff for optimization
        self.probability_model = probability_model

    def compare_kmers(self, i1, i2):
        pass

    def initialize(self, unique_kmers):
        pass

    def get_type(self):
        return self.metric_type

    def get_model_for_optimization(self):
        return self.probability_model


# UNIFIED INITIALIZER: probabilitiy model, background_info, additional_info
class GCcontent(KmerMetric):
    def __init__(self, probability_model, background_info, additional_info):
        super().__init__(name="gc",
                         metric_type="distance",
                         annot="absolute difference is GC content",
                         probability_model=probability_model
                         )
        self.gc = None

    def compare_kmers(self, i1, i2):
        a = tf.gather(self.gc, i1)
        b = tf.gather(self.gc, i2)
        return tf.abs(a - b)

    def initialize(self, unique_kmers):
        unique_kmers_tensor = tf.constant(unique_kmers)
        chars_tensor = tf.strings.bytes_split(unique_kmers_tensor)

        # Calculate GC content
        gc_content = tf.cast(
            tf.math.reduce_sum(tf.where((chars_tensor == b'G') | (chars_tensor == b'C'), 1, 0), axis=1), tf.float32)
        self.gc = gc_content


class PairContent(KmerMetric):
    def __init__(self, probability_model, background_info, additional_info):
        super().__init__(name="pair-gc",
                         metric_type="distance",
                         annot="absolute difference in nucleotide pair content",
                         probability_model=probability_model
                         )
        self.pairs = None
        self.info = ["".join(pair) for pair in product(list('ACGT'), list('ACGT'))]

    def __characterize(self, seq):
        seq_tensor = tf.strings.unicode_split(seq, 'UTF-8')

        # Generate all possible dinucleotide pairs and count occurrences
        def count_pairs(s):
            pairs = tf.strings.join([s[:-1], s[1:]], separator='')
            # Count occurrences of each pair in `self.info`
            counts = [tf.reduce_sum(tf.cast(tf.equal(pairs, pair), tf.int32)) for pair in self.info]
            return tf.stack(counts)

        res = count_pairs(seq_tensor)
        return res

    def compare_kmers(self, i1, i2):
        a = tf.gather(self.pairs, i1)
        b = tf.gather(self.pairs, i2)
        diff = a - b
        return tf.reduce_mean(tf.cast(diff * diff, tf.float32))

    def initialize(self, unique_kmers):
        unique_kmers_tensor = tf.constant(unique_kmers)
        self.pairs = tf.map_fn(self.__characterize, unique_kmers_tensor, dtype=tf.int32)
