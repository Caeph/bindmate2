import numpy as np
import pandas as pd
import swifter
from pyfastaq.sequences import file_reader as fasta_reader


def read_unique_kmers(input_sqs, k):
    input_sqs['kmers'] = input_sqs['sequence'].swifter.progress_bar(False).apply(
        lambda s: list(get_overlapping_kmers(s, k)))

    input_sqs['window'] = [[0, 1] for _ in range(len(input_sqs))]
    two_lined = input_sqs.reset_index().explode(['kmers', 'window'])
    two_lined['order'] = two_lined['kmers'].apply(lambda lst: np.arange(len(lst)))

    two_lined = two_lined.rename(columns={"index": "input_sequence_index"})

    kmers_mapped_to_sqs = two_lined.explode(['kmers', 'order']).drop(columns=['sequence']).reset_index().drop(
        columns=['index'])
    unique_kmers = kmers_mapped_to_sqs["kmers"].unique()
    mapper = {kmer: i for i, kmer in enumerate(unique_kmers)}

    kmers_mapped_to_sqs['kmer_index'] = kmers_mapped_to_sqs['kmers'].map(mapper)

    return unique_kmers, kmers_mapped_to_sqs


def load_fasta_input(filename, subset=None):
    df = pd.DataFrame([[entry.id, entry.seq] for entry in fasta_reader(filename)], columns=['header', 'sequence'])

    if subset is not None:
        df = df.sample(subset)
    return df


# def load_bed_input(filename, fasta_input_filename, subset=None):
#     ...
#     # TODO would be nice: load bed input to dataframe


def make_kmers(seq, k, offset):
    """
    Get k-mers from a seq with an offset. No sliding window, k-mers are exclusive.
    Only words of size k are returned (excess trimmed)
    :param seq:
    :param k:
    :param offset:
    :return:
    """
    excess = (len(seq) - offset) % k
    if excess != 0:
        s = seq[offset:-excess]
    else:
        s = seq[offset:]
    kmers = np.apply_along_axis(lambda x: "".join(x), 1, np.reshape(np.array(list(s)), (-1, k)))
    return kmers


def get_overlapping_kmers(seq, k, kmer_overlapping=2):
    part = k // kmer_overlapping
    offsets = [i * part for i in range(kmer_overlapping)]
    for offset in offsets:
        kmers = make_kmers(seq, k, offset)
        yield kmers
