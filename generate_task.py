import argparse
import os
import random

import yaml
import numpy as np

from tqdm import tqdm

#------------
# This file requires a new folder named "p-embeddings" to be created in the folder "..../preprocessed_data/[DatasetName]"
# The saved embeddings should be at "../preprocessed_data/LibriTTS/p-embeddings/task-0.npy"
#------------

class Generator:
    def __init__(self, config,
     task_size = 500, 
     support_set_size = 100, 
     query_set_size = 100, 
     valid_symbols = [],
     embedding_size = 1024
     ): # Final size of query set <= query_set_size
        self.valid_symbols = valid_symbols
        self.config = config
        self.task_size = task_size
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size
        self.embedding_size = embedding_size
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
        print("Creating phoneme coverage sets")
        for i, support_set in enumerate(tqdm(support_sets)) :
            for ind in support_set :
                phonemes = preprocessed_txt[ind].split("|")[2]
                phonemes = phonemes[1:-1].split(" ")
                for ph in phonemes :
                    phoneme_sets[i].add(ph)
        
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
        print("Deleting out of coverage samples")
        for i, query_set in enumerate(tqdm(query_sets)) :
            covered_list = []
            for ind in query_set :
                sentence_phonemes = set() # The set of phonemes that exist in a single sentence
                illegal = False 
                phonemes = preprocessed_txt[ind].split("|")[2]
                phonemes = phonemes[1:-1].split(" ")
                for ph in phonemes :
                    sentence_phonemes.add(ph)
                for ph in sentence_phonemes :
                    if ph not in phoneme_sets[i] : 
                        illegal = True
                if not illegal :
                    covered_list.append(ind)
            query_sets[i] = covered_list
        
        # Create phoneme coverage sets (I think this is optional?)
        query_phoneme_sets = [set() for _ in range(len(support_list))]
        print("Creating phoneme coverage sets")
        for i, query_set in enumerate(tqdm(query_sets)) :
            for ind in query_set :
                phonemes = preprocessed_txt[ind].split("|")[2]
                phonemes = phonemes[1:-1].split(" ")
                for ph in phonemes :
                    query_phoneme_sets[i].add(ph)
        
        # Assert phoneme coverage. The phonemes of the query set should be a subset of that of the corresponding support set.
        print("Checking phoneme coverage...")
        illegal = False
        for i, phonemes in enumerate(query_phoneme_sets) :
            for ph in phonemes :
                if ph not in phoneme_sets[i] :
                    illegal = True
        assert illegal == False
        print("Test passed")

        # Save query sets
        print("Saving query sets")
        f = open(os.path.join(self.out_dir, "query-list") + ".txt", "w+")
        for i in tqdm(range(self.task_size)) :
            line = "|".join([str(query_sets[i]), str(query_phoneme_sets[i]), str(len(query_phoneme_sets[i])), str(len(phoneme_sets[i]))]) # [List of indexes to form query set, the phonemes this set has, query phoneme coverage, support phoneme coverage]
            f.write(line + "\n")
    
    # --This function has bad readability. I should be more careful about naming--
    def build_p_embeddings(self) :
        # Read training data
        preprocessed_txt = open(os.path.join(self.in_dir, self.train_set) + ".txt", 'r')
        preprocessed_txt = preprocessed_txt.readlines()
        print("Support set pool size :", len(preprocessed_txt))

        # Read the list of support sets created from build_support_sets
        print("Reading support sets")
        support_list = open(os.path.join(self.in_dir, "support-list") + ".txt", 'r')
        support_list = support_list.readlines()
        print("Number of support sets :", len(support_list))
        print("Size of support set :", len(support_list[0].split("|")[0].split(",")))

        # Parse support set indexes
        indexes = []
        for support_set in support_list :
            set_list = support_set.split("|")[0][1:-1].split(",")
            indexes.append(set_list)
        
        # Generate file names for each set
        file_names = []
        for index_list in indexes :
            file_list = []
            for ind in index_list :
                file_name = preprocessed_txt[int(ind)].split("|")[1] + "-representation-" + preprocessed_txt[int(ind)].split("|")[0] + ".npy"
                file_list.append(file_name)
            file_names.append(file_list)
        
        # --This part is hard to understand since the naming is chaotic and the operations aren't really optimized. Could be improved but it works, I think--
        #Calculate and save the average representation of every phoneme
        print("Calculating and saving the average representation of every phoneme")
        print("Final shape of phoneme embeddings : " + "(" + str(len(self.valid_symbols)) + ", " + str(self.embedding_size) + ")")
        for k, file_list in enumerate(tqdm(file_names)) :
            table = dict()
            table_count = dict()
            table_save = []
            for s in self.valid_symbols :
                table[s] = np.zeros(self.embedding_size)
                table_count[s] = 0
            for i, file_name in enumerate(file_list) :
                representation = np.load(os.path.join(self.in_dir, "representation", file_name))
                id_in_preprocessed_text = indexes[k][i]
                appeared_phonemes = preprocessed_txt[int(id_in_preprocessed_text)].split("|")[2][1:-1].split(" ")
                for i, ph in enumerate(appeared_phonemes) :
                    table[ph] = table[ph] + representation[i] # Can't use index function since it always chooses the first occurance.
                    table_count[ph] = table_count[ph] + 1
            for key in table.keys() :
                if table_count[key] != 0 :
                    table[key] = table[key] / table_count[key]
                table_save.append(table[key])
            table_save = np.array(table_save)
            np.save(os.path.join(self.out_dir, "p-embeddings", "task-" + str(k)), table_save)
        print("All embeddings saved at", os.path.join(self.out_dir, "p-embeddings"))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', "sil", "sp", "spn"
    ] # This is for English. This code is bad and temporary.

    generator = Generator(config, valid_symbols=valid_symbols)

    #generator.build_support_sets() # Uncomment this line to generate the support set first.
    #generator.build_query_sets() # Build the query set based on the generated support set
    generator.build_p_embeddings() # Build the embeddings based on the support sets. Doesn't require query set(aka build_query_sets).