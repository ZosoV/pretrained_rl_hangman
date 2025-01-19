import torch
from torch.nn import Module
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from torch.utils.data import Dataset, DataLoader

# Custom Character-Level Tokenizer
class CharTokenizer:
    def __init__(self):
        self.vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz", start=1)}
        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = len(self.vocab)
        self.vocab["[CLS]"] = len(self.vocab)
        self.vocab["[MASK]"] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print("Vocab size: ", len(self.vocab))

    def tokenize(self, text):
        return list(text.lower())
    
    def get_ids(self, tokens):
        return [self.vocab.get(t, self.vocab["[UNK]"]) for t in tokens]

    def encode(self, text, max_length):
        tokens = self.tokenize(text)
        token_ids = self.get_ids(tokens)
        return [self.vocab["[CLS]"]] + token_ids[:max_length] + [self.vocab["[PAD]"]] * (max_length - len(token_ids) - 1)
    
    def decode(self, token_ids):
        return ''.join([self.inv_vocab.get(t, "[UNK]") for t in token_ids if t != 0])
    
# Custom Dataset
class WordDataset(Dataset):
    def __init__(self, words, tokenizer, max_length):
        self.words = words
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)
    
    def mlm_masking_word(self, sentence):
        # Tokenize the entire sentence
        # tokenized = self.tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        # tokens = tokenized["input_ids"].squeeze(0)  # Shape: (seq_len,)

        tokens = self.tokenizer.tokenize(sentence)
        token_ids = torch.tensor(self.tokenizer.get_ids(tokens))

        # Generate random probabilities for each token
        probs = torch.rand(token_ids.shape)

        # 15% of the tokens will be considered for masking
        mask_prob = probs < 0.15

        # If no token is masked, select a random token to mask
        if not mask_prob.any():
            mask_prob[torch.randint(0, token_ids.shape[0], (1,))] = True
        
        # print("0.15 Masked:",  mask_prob)

        # Initialize labels (original tokens for masked positions, 0 otherwise)
        labels = torch.where(mask_prob, token_ids, torch.zeros_like(token_ids))
        # print("labels:", labels)

        # 80% of masked tokens will be replaced with [MASK]
        mask_replace_prob = torch.rand(token_ids.shape)
        masked_tokens = torch.where(
            mask_prob & (mask_replace_prob < 0.8), 
            torch.tensor(self.tokenizer.vocab['[MASK]']), 
            token_ids
        )
        # print("80% from masked: ", mask_prob & (mask_replace_prob < 0.8))
        # print(masked_tokens)

        # 10% of masked tokens will be replaced with random tokens
        random_replace_prob = torch.rand(token_ids.shape)
        random_tokens = torch.randint(len(self.tokenizer.vocab), token_ids.shape)
        final_tokens = torch.where(
            mask_prob & (mask_replace_prob >= 0.8) & (random_replace_prob < 0.5),
            random_tokens,
            masked_tokens
        )
        # print("10% from masked: ", mask_prob & (mask_replace_prob >= 0.8) & (random_replace_prob < 0.5))
        # print(final_tokens)

        # Tokens not selected for masking remain unchanged
        # final_tokens = torch.where(mask_prob, final_tokens, token_ids)
        # print(final_tokens)

        # Adding special tokens ids and correcting labels
        return self.add_special_tokens(final_tokens, labels)
    

    def add_special_tokens(self, token_ids, labels):
        # Create CLS and PAD tokens
        cls_token = torch.tensor([self.tokenizer.vocab["[CLS]"]])
        pad_token = torch.tensor([self.tokenizer.vocab["[PAD]"]])

        # Add CLS token and truncate or pad token_ids
        truncated_tokens = token_ids[:self.max_length]
        padded_tokens = torch.cat([cls_token, truncated_tokens, pad_token.repeat(self.max_length - truncated_tokens.size(0) - 1)])

        # Add 0 for CLS and PAD tokens to labels
        zero_label = torch.tensor([0])
        truncated_labels = labels[:self.max_length]
        padded_labels = torch.cat([zero_label, truncated_labels, zero_label.repeat(self.max_length - truncated_labels.size(0) - 1)])

        # Outputs
        final_tokens = padded_tokens  # Shape: (max_length,)
        labels = padded_labels         # Shape: (max_length,)
        return final_tokens, labels


    def __getitem__(self, idx):
        word = self.words[idx]

        input_ids, labels = self.mlm_masking_word(word)

        attention_mask = torch.where(input_ids != self.tokenizer.vocab["[PAD]"], 1, 0)

        output = {"bert_input": input_ids,
                  "bert_label": labels,
                  "attention_mask": attention_mask}
        
        return output
    
