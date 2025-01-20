import gymnasium as gym
from gymnasium import spaces

import numpy as np
from collections import Counter, defaultdict
import string

class HangmanEnv(gym.Env):
    def __init__(self, word_list):
        super(HangmanEnv, self).__init__()

        self.word_list = word_list
        self.vocab = string.ascii_lowercase
        self.max_lives = 6


        self.top_bigrams, self.top_trigrams = self._calculate_top_n_gram()

        # Observation space: Current word state and one-hot vector for tried letters
        self.observation_space = spaces.Dict({
            "word_state": spaces.MultiDiscrete([len(self.vocab) + 1] * 25),
            "tried_letters": spaces.MultiBinary(len(self.vocab))
        })

        # Action space: Choosing a letter from the vocabulary
        self.action_space = spaces.Discrete(len(self.vocab))

        # Precompute global letter frequencies
        letter_counts = Counter("".join(self.word_list))
        total_letters = sum(letter_counts.values())
        self.global_letter_frequencies = {
            letter: count / total_letters for letter, count in letter_counts.items()
        }

    def reset(self, word_id=None):
        self.target_word = np.random.choice(self.word_list)
        if word_id is not None:
            self.target_word = self.word_list[word_id]
        self.word_state = ["_"] * len(self.target_word)
        self.tried_letters = set()
        self.lives = self.max_lives
        
        return self._get_obs()

    def step(self, action):
        letter = self.vocab[action]
        reward = 0

        if letter in self.tried_letters:
            reward = -2  # Penalty for repeating a letter
        else:
            self.tried_letters.add(letter)
            if letter in self.target_word:
                # Correct guess: Update word state and calculate reward
                reward += self._calculate_correct_guess_reward(letter)
                for i, char in enumerate(self.target_word):
                    if char == letter:
                        self.word_state[i] = letter

                # Check if the game is won
                if "_" not in self.word_state:
                    return self._get_obs(), 50, True, {"result": "win"}
            else:
                # Incorrect guess: Deduct a life and apply penalty
                self.lives -= 1
                reward -= 5

                # Check if the game is lost
                if self.lives <= 0:
                    return self._get_obs(), reward, True, {"result": "lose"}

        return self._get_obs(), reward, False, {}
    
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


    def _calculate_correct_guess_reward(self, letter, scale=1):
        # Reward for correct guess based on global and relative frequencies
        global_reward = self.global_letter_frequencies.get(letter, 0)
        
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

        return global_reward + relative_reward + bigram_bonus + trigram_bonus

    def _get_obs(self):
        # Encode word state and tried letters as observations
        word_state_encoded = [
            self.vocab.index(char) if char in self.vocab else len(self.vocab)
            for char in self.word_state
        ] + [len(self.vocab)] * (25 - len(self.word_state))
        
        tried_letters_encoded = [1 if char in self.tried_letters else 0 for char in self.vocab]

        return {
            "word_state": np.array(word_state_encoded, dtype=np.int32),
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

    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()
        # Map action to letter
        print(f"Action: {id2char[action]} Reward: {reward}")

    print("Game Over:", info)
