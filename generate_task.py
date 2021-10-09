import argparse
import os
import random

import yaml

from tqdm import tqdm

class Generator:
    def __init__(self, config, task_size = 500, support_set_size = 100, query_set_size = 100): # Final size of query set <= query_set_size
        self.config = config
        self.task_size = task_size
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size
        self.in_dir = config["path"]["preprocessed_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        if "subsets" in config:
            self.train_set = config["subsets"].get("train", None)
            self.val_set = config["subsets"].get("val", None) # val set is equivalent to dev set, I think.
            self.test_set = config["subsets"].get("test", None)
        
        random.seed(69) # For the reproducibility of generating tasks

    def build_support_sets(self) :
        # Read training data
        preprocessed_txt = open(os.path.join(self.in_dir, self.train_set) + ".txt", 'r')
        preprocessed_txt = preprocessed_txt.readlines()
        print("Support set pool size :", len(preprocessed_txt))

        # Generate the support sets for (task_size) amount of tasks
        support_sets = []
        for i in range(self.task_size) : 
            sample_list = random.sample(range(len(preprocessed_txt)), self.support_set_size)
            support_sets.append(sample_list)

        # Create phoneme coverage sets
        phoneme_sets = [set() for _ in range(self.task_size)]
        index = 0
        print("Creating phoneme coverage sets")
        for support_set in tqdm(support_sets) :
            for ind in support_set :
                phonemes = preprocessed_txt[ind].split("|")[2]
                phonemes = phonemes[1:-1].split(" ")
                for ph in phonemes :
                    phoneme_sets[index].add(ph)
            index = index + 1
        
        # Save support sets
        print("Saving support sets")
        f = open(os.path.join(self.out_dir, "support-list") + ".txt", "w+")
        for i in tqdm(range(self.task_size)) :
            line = "|".join([str(support_sets[i]), str(phoneme_sets[i]), str(len(phoneme_sets[i]))]) # [List of indexes to form support set, the phonemes this set has, the phoneme coverage in number]
            f.write(line + "\n")
    
    def build_query_sets(self) :
        # Read the list of support sets created from build_support_sets
        print("Reading support sets")
        support_list = open(os.path.join(self.in_dir, "support-list") + ".txt", 'r')
        support_list = support_list.readlines()
        print("Number of support sets :", len(support_list))
        print("Size of support set :", len(support_list[0].split("|")[0].split(",")))

        # Parse phoneme coverages from support sets
        phoneme_sets = []
        for line in support_list :
            phonemes = line.split("|")[1]
            phonemes = phonemes[1:-1].replace("'", "").replace(" ", "").split(",")
            phoneme_sets.append(set(phonemes))

        # Read dev data
        preprocessed_txt = open(os.path.join(self.in_dir, self.val_set) + ".txt", 'r')
        preprocessed_txt = preprocessed_txt.readlines()
        print("Query set pool size :", len(preprocessed_txt))

        # Generate the query sets for len(support_list) amount of tasks. len(support_list) should equal task_size
        query_sets = []
        random.seed(420)
        for i in range(self.task_size) : 
            sample_list = random.sample(range(len(preprocessed_txt)), self.query_set_size)
            query_sets.append(sample_list)
        
        # Delete out of coverage samples
        index = 0
        print("Deleting out of coverage samples")
        for query_set in tqdm(query_sets) :
            covered_list = []
            for ind in query_set :
                sentence_phonemes = set() # The set of phonemes that exist in a single sentence
                illegal = False 
                phonemes = preprocessed_txt[ind].split("|")[2]
                phonemes = phonemes[1:-1].split(" ")
                for ph in phonemes :
                    sentence_phonemes.add(ph)
                for ph in sentence_phonemes :
                    if ph not in phoneme_sets[index] : 
                        illegal = True
                if not illegal :
                    covered_list.append(ind)
            query_sets[index] = covered_list
            index = index + 1
        
        # Create phoneme coverage sets (I think this is optional?)
        query_phoneme_sets = [set() for _ in range(len(support_list))]
        index = 0
        print("Creating phoneme coverage sets")
        for query_set in tqdm(query_sets) :
            for ind in query_set :
                phonemes = preprocessed_txt[ind].split("|")[2]
                phonemes = phonemes[1:-1].split(" ")
                for ph in phonemes :
                    query_phoneme_sets[index].add(ph)
            index = index + 1
        
        # Assert phoneme coverage. The phonemes of the query set should be a subset of that of the corresponding support set.
        print("Checking phoneme coverage...")
        illegal = False
        index = 0
        for phonemes in query_phoneme_sets :
            for ph in phonemes :
                if ph not in phoneme_sets[index] :
                    illegal = True
            index = index + 1
        assert illegal == False
        print("Test passed")

        # Save query sets
        print("Saving query sets")
        f = open(os.path.join(self.out_dir, "query-list") + ".txt", "w+")
        for i in tqdm(range(self.task_size)) :
            line = "|".join([str(query_sets[i]), str(query_phoneme_sets[i]), str(len(query_phoneme_sets[i])), str(len(phoneme_sets[i]))]) # [List of indexes to form query set, the phonemes this set has, query phoneme coverage, support phoneme coverage]
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    generator = Generator(config)
    #generator.build_support_sets() # Uncomment this line to generate the support set first.
    generator.build_query_sets()