import numpy as np
import tensorflow as tf
from itertools import product
from collections import Counter
import pyProBound
from background_normalizer import *


class KmerMetric:
    def __init__(self,
                 name, annot, metric_type,
                 probability_model, additional_info, unique_kmers
                 ):
        self.name = name
        self.annotation = annot
        self.metric_type = metric_type  # distance or similarity

        # probability_functions and stuff for optimization
        self.probability_model = probability_model

        # background
        if "background_info" in additional_info:
            self.normalizer = Normalizer(additional_info["background_info"], additional_info['k'])
        elif "background_object" in additional_info:
            self.normalizer = additional_info["background_object"]
        else:
            self.normalizer = Normalizer(None, additional_info['k'])
            # dummy object that does nothing
        self.unique_kmers = unique_kmers

    def compare_kmers(self, combinations):
        pass

    def get_type(self):
        return self.metric_type

    def get_model_for_optimization(self):
        return self.probability_model


# UNIFIED INITIALIZER: probabilitiy model, background_info, additional_info
class GCcontent(KmerMetric):
    def __init__(self, probability_model, additional_info, unique_kmers):
        super().__init__(name="gc",
                         metric_type="distance",
                         annot="absolute difference is GC content",
                         probability_model=probability_model,
                         additional_info=additional_info,
                         unique_kmers=unique_kmers
                         )
        self.gc = None

    def compare_kmers(self, combinations):
        unique_kmers_tensor = tf.constant(self.unique_kmers)

        def gc_content_diff(kmers_tensor):
            chars_tensor = tf.strings.bytes_split(kmers_tensor)
            # Calculate GC content
            gc_content = tf.cast(
                tf.math.reduce_sum(tf.where((chars_tensor == b'G') | (chars_tensor == b'C'), 1, 0), axis=1), tf.float32)

            hor, ver = tf.meshgrid(gc_content, gc_content, indexing='ij')
            diff2d = tf.math.abs(hor - ver)
            diff1d = tf.reshape(diff2d, [-1])
            return diff1d

        diff = gc_content_diff(unique_kmers_tensor)

        diff = self.normalizer.normalize(gc_content_diff, diff)
        return diff


class PairContent(KmerMetric):
    def __init__(self, probability_model, additional_info, unique_kmers):
        super().__init__(name="pair-gc",
                         metric_type="distance",
                         annot="absolute difference in nucleotide pair content",
                         probability_model=probability_model,
                         additional_info=additional_info,
                         unique_kmers=unique_kmers
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

    def compare_kmers(self, combinations):
        kmers_tensor = tf.constant(self.unique_kmers)

        def nucl_pair_diff(unique_kmers_tensor):
            pairs = tf.map_fn(self.__characterize, unique_kmers_tensor, dtype=tf.int32)
            full_difference = tf.zeros((len(pairs), len(pairs)), dtype=tf.float64)
            for i in range(pairs.shape[1]):
                a = pairs[:, i]
                b = pairs[:, i]
                a_rep, b_rep = tf.meshgrid(a, b, indexing='ij')
                diff = tf.cast(a_rep - b_rep, tf.float64)
                full_difference += diff * diff

            diff = tf.reshape(full_difference, [-1])
            return diff

        mse = nucl_pair_diff(kmers_tensor)
        mse = self.normalizer.normalize(nucl_pair_diff, mse)
        return mse


class ProBoundAffinity(KmerMetric):
    def __init__(self, probability_model, additional_info):
        super().__init__(name="probound",
                         metric_type="distance",
                         annot="difference in affinity to TF selection as modelled by ProBound",
                         probability_model=probability_model,
                         additional_info=additional_info
                         )

    def compare_kmers(self, combinations):
        ...

    def initialize(self, unique_kmers):
        ...
