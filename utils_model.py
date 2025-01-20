import torch
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers import BertConfig, BertModel, BertTokenizer
from torch.nn import Module
import numpy as np


# Load char embeddings
def load_char_embeddings(embeddings_path):
    print('Processing pretrained character embeds...')
    char_embeddings = {}
    with open(embeddings_path, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            vec = torch.tensor(np.array(line_split[1:], dtype=float))
            char = line_split[0]
            char_embeddings[char] = vec

    # For mask use underscore
    token_mask = char_embeddings["_"]

    # For padding use point
    token_pad = char_embeddings["."]

    # For CLS used to return an cumulated information for the whole word use #
    token_cls = char_embeddings["#"]

    # Filter Character Embeddings for char from a-z and three special tokens
    char_embeddings = {char: vec for char, vec in char_embeddings.items() if char in "abcdefghijklmnopqrstuvwxyz"}
    # Assing char indeces from 0 to 25 to the letters and 26 to 28 to the special tokens
    char_indices = {char: i for i, char in enumerate(char_embeddings.keys(), start=1)}

    char_embeddings["."] = token_pad
    char_embeddings["?"] = token_mask
    char_embeddings["#"] = token_cls
    char_embeddings["_"] = token_mask

    char_indices["."] = 0 #  PAD
    char_indices["?"] = 26 # UNK
    char_indices["#"] = 27 # CLS
    char_indices["_"] = 28 # MASK

    # Create an embedding matrix E
    embedding_matrix = torch.zeros((len(char_embeddings), 300))
    #embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
    for char, i in char_indices.items():
        #print ("{}, {}".format(char, i))
        embedding_vector = char_embeddings.get(char)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Head using embedding layer
# If I use the 300 Glove embeddings I can use the embedding layer to predict the word
class MLMHeadE(torch.nn.Module):
    def __init__(self, embedding_layer):
        """
        :param embedding_layer: Embedding layer from the model
        """
        super().__init__()
        # Use the embedding layer's weight matrix for the linear layer
        self.linear = torch.nn.Linear(embedding_layer.word_embeddings.weight.size(1),
                                       embedding_layer.word_embeddings.weight.size(0))
        self.linear.weight = embedding_layer.word_embeddings.weight  # Share weights
        self.softmax = torch.nn.LogSoftmax(dim=-1)        

    def forward(self, x):
        return self.softmax(self.linear(x))

class MLMHead(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

# Custom BERT Architecture with Configurable Layers
class CustomBERT(Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, 
                 max_position_embeddings, intermediate_size,
                 embeddings_path = "data/glove.840B.300d-char.txt"):
        super(CustomBERT, self).__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size,
        )
        self.embeddings = BertEmbeddings(self.config)
        self.encoder = BertEncoder(self.config)

        self.mlm_head = MLMHead(hidden_size, vocab_size)
        # if hidden_size == 300:
        #     print("Replacing embedding weights and MLP head")
        #     self.replace_embedding_weights(embeddings_path)
        # self.mlm_head = MLMHeadE(self.embeddings)
        # self.pooler = BertPooler(config)

    def replace_embedding_weights(self, embeddings_path):
        # Load the embeddings
        embedding_matrix = load_char_embeddings(embeddings_path)
        # Replace the weights of the embedding layer
        self.embeddings.word_embeddings.weight = torch.nn.Parameter(embedding_matrix)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids=input_ids)

        # NOTE: I have to add to dimension in between for the attention mask
        # because it will be used to calculatation the attention scores
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Encoder ouputs can return the embeddings in each layer, but here
        # I only interested in the last hidden state
        encoder_outputs = self.encoder(embeddings, attention_mask=attention_mask, return_dict=True)

        # Pooler is used to get the CLS token embedding and apply 
        # a linear transformation to it + tanh activation
        # output = self.pooler(encoder_outputs.last_hidden_state)

        # MLM head output
        output = self.mlm_head(encoder_outputs.last_hidden_state)

        # return encoder_outputs.last_hidden_state, output
        return output