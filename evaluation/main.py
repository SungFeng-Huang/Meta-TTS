from argparse import ArgumentParser
from wavs_to_dvector import WavsToDvector
from centroid_similarity import CentroidSimilarity
from pair_similarity import PairSimilarity
from speaker_verification import SpeakerVerification


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--new_pair', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default='eer.txt')
    args = parser.parse_args()
    main = WavsToDvector(args)

    main = CentroidSimilarity()
    main.load_dvector()
    main.get_centroid_similarity()
    main.save_centroid_similarity()

    main = PairSimilarity()
    main.load_dvector()
    main.get_pair_similarity()
    main.save_pair_similarity()

    main = SpeakerVerification(args)
    main.load_pair_similarity()
    main.get_eer()
    # for suffix in [ '_encoder']:
    # for suffix in ['_base_emb', '_base_emb1', '_meta_emb', '_meta_emb1']:
        # main.set_suffix(suffix)
        # main.plot_eer(suffix)
        # main.plot_auc(suffix)

