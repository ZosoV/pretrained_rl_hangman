import gymnasium as gym
from gymnasium import spaces

import numpy as np
from collections import Counter, defaultdict
import string
from transformers import BertTokenizer

class HangmanEnv(gym.Env):
    def __init__(self, word_list, max_length_word = 32, pretrained_bert = False):
        super(HangmanEnv, self).__init__()

        self.word_list = word_list
        self.vocab = string.ascii_lowercase
        self.max_lives = 6
        self.max_length_word = max_length_word
        self.pretrained_bert = pretrained_bert

        # Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # To match the tokenizer identifiers set the special tokens id
        self.mask_token = 29
        self.cls_token = 28
        self.pad_token = 0 

        self.top_bigrams, self.top_trigrams = self._calculate_top_n_gram()

        # Observation space: Current word state and one-hot vector for tried letters
        # spaces.Box( low = 0, high = np.inf, shape=(4,), dtype=np.float64)
        # self.observation_space = spaces.Dict({
        #     "observation": spaces.MultiDiscrete([len(self.vocab) + 2] * max_length_word),
        #     "tried_letters": spaces.MultiBinary(len(self.vocab))
        # })
        # NOTE: sum +4 to the high because there are 4 special tokens in the tokenizer
        # UNK, PAD, CLS, MASK
        self.observation_space = spaces.Dict({
            "observation": spaces.Box( low = 0, high = len(self.vocab) + 4, shape=(self.max_length_word,), dtype=np.int32),
            "tried_letters": spaces.Box( low = 0, high = 1, shape=(len(self.vocab),), dtype=np.int32)
        })

        # Action space: Choosing a letter from the vocabulary
        self.action_space = spaces.Discrete(len(self.vocab))

        # Precompute global letter frequencies
        self._precompute_global_letter_frequencies()
        self._precompute_relative_letter_frequencies_by_length()

    def _precompute_global_letter_frequencies(self):
        letter_counts = Counter("".join(self.word_list))
        total_letters = sum(letter_counts.values())
        self.global_letter_frequencies = {
            letter: count / total_letters for letter, count in letter_counts.items()
        }

    def _precompute_relative_letter_frequencies_by_length(self):
        self.relative_letter_frequencies = {}
        for length in range(1, self.max_length_word + 1):
            matching_words = [w for w in self.word_list if len(w) == length]
            relative_letter_counts = Counter("".join(matching_words))
            total_relative_letters = sum(relative_letter_counts.values())
            self.relative_letter_frequencies[length] = {
                l: (relative_letter_counts.get(l, 0) + 1) / (total_relative_letters + len(self.vocab))
                for l in self.vocab }


    def _calculate_top_n_gram(self, n = 50):
        # Calculate all the bigrams and trigrams frequencies in the wordlist
        bigram_frequencies = defaultdict(int)

        for word in self.word_list:
            for i in range(1, len(word)):
                bigram_frequencies[word[i - 1] + word[i]] += 1

        trigram_frequencies = defaultdict(int)

        for word in self.word_list:
            for i in range(2, len(word)):
                trigram_frequencies[word[i - 2] + word[i - 1] + word[i]] += 1

        # Divide to the total number of bigrams and trigrams
        total_bigrams = sum(bigram_frequencies.values())
        total_trigrams = sum(trigram_frequencies.values())

        bigram_frequencies = {k: v / total_bigrams for k, v in bigram_frequencies.items()}
        trigram_frequencies = {k: v / total_trigrams for k, v in trigram_frequencies.items()}

        # Take the top 50 bigrams and trigrams
        top_bigrams = dict(sorted(bigram_frequencies.items(), key=lambda item: item[1], reverse=True)[:n])
        top_trigrams = dict(sorted(trigram_frequencies.items(), key=lambda item: item[1], reverse=True)[:n])

        return list(top_bigrams.keys()), list(top_trigrams.keys())
    
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Select a random word from the word list using self.np_random
        random_int = self.np_random.integers(0, len(self.word_list), dtype=int)
        self.target_word = self.word_list[random_int]

        if options and "word_id" in options:
            self.target_word = self.word_list[options['word_id']]
        self.word_state = ["_"] * len(self.target_word)
        self.tried_letters = set()
        self.lives = self.max_lives
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        letter = self.vocab[action]
        reward = 0

        info_bonus = {}

        if letter in self.tried_letters:
            reward = -2  # Penalty for repeating a letter
            self.lives -= 1 # NOTE: I added this line to penalize the player for repeating a letter
                            # I haven't check experiments with this line yet.
        else:
            self.tried_letters.add(letter)
            if letter in self.target_word:
                # Correct guess: Update word state and calculate reward
                bonus, info_bonus = self._calculate_correct_guess_reward(letter)
                # NOTE: a reward of 10 for guessing
                reward += 10 + bonus
                for i, char in enumerate(self.target_word):
                    if char == letter:
                        self.word_state[i] = letter

                # Check if the game is won
                if "_" not in self.word_state:
                    info_bonus["result"] = "win"
                    return self._get_obs(), 50, True, False, info_bonus
            else:
                # Incorrect guess: Deduct a life and apply penalty
                self.lives -= 1
                reward -= 5

                # Check if the game is lost
                if self.lives <= 0:
                    return self._get_obs(), reward, True, False, {"result": "loss"}

        return self._get_obs(), reward, False, False, info_bonus


    def _calculate_correct_guess_reward(self, letter, scale=10):
        # Reward for correct guess based on global and relative frequencies
        global_reward = self.global_letter_frequencies.get(letter, 0) * scale
        
        # Calculate relative frequencies for current pattern
        pattern = "".join(
            c if c != "_" else "." for c in self.word_state
        )
        matching_words = [w for w in self.word_list if len(w) == len(self.target_word) and all(
            p == c or p == "." for p, c in zip(pattern, w))]
        relative_letter_counts = Counter("".join(matching_words))
        total_relative_letters = sum(relative_letter_counts.values())
        relative_frequencies = {
            l: (relative_letter_counts.get(l, 0) + 1) / (total_relative_letters + len(self.vocab))
            for l in self.vocab
        }

        relative_reward = relative_frequencies[letter] * scale

        # Additional reward for matching bigrams or trigrams with the current letter

        temp_dict = {k : 0.5 for k in self.top_bigrams}
        bigram_bonus = sum(
            temp_dict.get(self.word_state[i - 1] + letter, 0)
            for i in range(1, len(self.word_state)) if self.word_state[i - 1] != "_"
        )

        temp_dict = {k : 1. for k in self.top_trigrams}
        trigram_bonus = sum(
            temp_dict.get("".join(self.word_state[0:2]) + letter, 0)
            for i in range(2, len(self.word_state)) if "".join(self.word_state[0:2]) != "__"
        )

        # Calculate a ratio depending if the game is starting or ending
        # Count the number of underscores in the word state
        num_underscores = self.word_state.count("_")

        # Set the ratio as num_underscores / len(self.word_state)
        ratio = num_underscores / len(self.word_state)

        # NOTE: if the ratio is close to 0 the game is ending
        # if the ratio is close to 1 the game is starting
        # 
        # if the game is starting prioritize the global reward
        # if the game is ending prioritize the relative reward        

        # global_reward *= ratio
        # relative_reward *= 1 - ratio
        info_rewards = {
            "global_reward": global_reward,
            "relative_reward": relative_reward,
            # "bigram_bonus": bigram_bonus,
            # "trigram_bonus": trigram_bonus
        }

        return global_reward * ratio + relative_reward * (1 - ratio), info_rewards # + bigram_bonus + trigram_bonus, info_rewards # 

    def _get_obs(self):

        def small_bert_tokenizer(word):
            # Encode word state as integers (numeric representation only)
            # NOTE: sum  + 1 to shifht the indexes because the 0 is for the padding
            word_state_encoded = [
                self.vocab.index(char) + 1 if char in self.vocab else self.mask_token  # self.mask_token for '_'
                for char in word
            ]

            # Add cls token in the beginning
            word_state_encoded = [self.cls_token] + word_state_encoded

            # Pad word_state_encoded to match max_length_word
            word_state_encoded += [self.pad_token] * (self.max_length_word - len(word_state_encoded))
            return word_state_encoded
        
        def pretrained_bert_tokenizer(masked_word, max_length=32):
            masked_word = " ".join(masked_word)

            masked_word = masked_word.replace('_', '[MASK]')

            batch_info = self.tokenizer(masked_word, truncation=True, padding='max_length', return_tensors="np", max_length=max_length)

            return batch_info["input_ids"].astype(np.int32).squeeze()
        
        # NOTE: if using large model
        if self.pretrained_bert:
            word_state_encoded = pretrained_bert_tokenizer(self.word_state, self.max_length_word)
        else:
            word_state_encoded = small_bert_tokenizer(self.word_state)

        # NOTE: these are not shifted because this will be used for the one-hot encoding
        tried_letters_encoded = [1 if char in self.tried_letters else 0 for char in self.vocab]

        return {
            "observation": np.array(word_state_encoded, dtype=np.int32),
            "tried_letters": np.array(tried_letters_encoded, dtype=np.int32)
        }

    def render(self, mode="human"):
        if mode == "human":
            print("Word:", " ".join(self.word_state))
            print("Tried Letters:", " ".join(sorted(self.tried_letters)))
            print(f"Lives Left: {self.lives}")

    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    word_list = ["apple", "banana", "cherry", "date"]
    id2char = {i: c for i, c in enumerate(string.ascii_lowercase)}

    env = HangmanEnv(word_list)
    obs = env.reset()
    env.render()

    # NOTE: Here is a little bit of a miss match the possible action are values from 0 to 25
    # but the indices in the observation are from 1 to 26 in observation because it is prepared
    # to go into the BERT model
    # additionally special token are added to the observation
    # PAD: 0, UNK: 27, CLS: 28, MASK: 29
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncate, info = env.step(action)
        env.render()
        # Map action to letter
        print(f"Observation: {obs}")
        print(f"Done: {done}, Info: {info}")
        print(f"Action: {id2char[action]} Reward: {reward}")

    print("Game Over:", info)