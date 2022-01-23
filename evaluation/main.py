from argparse import ArgumentParser
from wavs_to_dvector import WavsToDvector
from centroid_similarity import CentroidSimilarity


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--new_pair', type=bool, default=False)
    args = parser.parse_args()
    main = WavsToDvector(args)
    #main.get_dvector()
    main = CentroidSimilarity()
    main.load_dvector()
    main.get_centroid_similarity()
    main.save_centroid_similarity()
