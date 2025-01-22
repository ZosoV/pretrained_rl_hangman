import torch
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers import BertConfig, BertModel, BertTokenizer
from torch.nn import Module
import numpy as np
from transformers import BertPreTrainedModel

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
        # Sum 26 to feed the tried letters
        # self.linear = torch.nn.Linear(hidden + 26, vocab_size)
        self.linear = torch.nn.Linear(hidden, vocab_size)

        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, tried_letter = None):
        # if tried_letter.dim() == 1:
        #     tried_letter = tried_letter.unsqueeze(0)
        #     x = torch.concat((x, tried_letter), dim=1)
        x = self.linear(x)

        return self.softmax(x)
    
class DQNHead(torch.nn.Module):
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
        # NOTE: I add 26 to handle a vector of tried letters
        # NOTE: vocab_size it's counting the special tokens too here
        # I don't want that
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden + 26, 512),  # Hidden size + tried letters size
            torch.nn.ReLU(),
            torch.nn.Linear(512, vocab_size - 4)  # Removing special tokens
        )

    def forward(self, x, tried_letter):
        # Concatenate the tried letter to the last hidden state
        if tried_letter.dim() == 1:
            tried_letter = tried_letter.unsqueeze(0)
        x = torch.concat((x, tried_letter), dim=1)

        return self.classifier(x)

# Custom BERT Architecture with Configurable Layers
class CustomBERT(BertPreTrainedModel):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, 
                 max_position_embeddings, intermediate_size,
                #  embeddings_path = "data/glove.840B.300d-char.txt",
                 dqn_head = False):
        config = BertConfig(
            # vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size,
        )
        super().__init__(config)

        # Load the pretrained BERT model
        self.bert = BertModel(config)


        # self.embeddings = BertEmbeddings(self.config)
        # self.encoder = BertEncoder(self.config)

        self.head = MLMHead(hidden_size, vocab_size)
        self.dqn_head = dqn_head
        if dqn_head:
            self.head = DQNHead(hidden_size, vocab_size)
        # if hidden_size == 300:
        #     print("Replacing embedding weights and MLP head")
        #     self.replace_embedding_weights(embeddings_path)
        # self.mlm_head = MLMHeadE(self.embeddings)
        # self.pooler = BertPooler(config)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        print("CustomBERT number of parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

        # Initialize weights
        self.init_weights()

    def replace_embedding_weights(self, embeddings_path):
        # Load the embeddings
        embedding_matrix = load_char_embeddings(embeddings_path)
        # Replace the weights of the embedding layer
        self.embeddings.word_embeddings.weight = torch.nn.Parameter(embedding_matrix)

    def _get_random_masked_token_mask(self, input_ids):
        mask = (input_ids == self.tokenizer.mask_token_id)

        # Create random values for tie-breaking
        random_values = torch.rand_like(input_ids.float())

        # Set random values only where the mask is True
        random_values[~mask] = float('-inf')  # Set irrelevant positions to -inf

        # Find the index of the maximum random value per row
        _, selected_indices = random_values.max(dim=1)

        # Create the final mask
        final_mask = torch.zeros_like(mask, dtype=torch.bool)
        final_mask[torch.arange(mask.size(0)), selected_indices] = True

        return final_mask

    def forward(self, input_ids, tried_letter = None, attention_mask = None):

        # NOTE: The collector uses a tensor for checking something
        # but this tensor is not in batch format, if not added the batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is None:
            # attention_mask = torch.where(input_ids != self.tokenizer.vocab["[PAD]"], 1, 0)
            # NOTE: the padding has index 0
            attention_mask = torch.where(input_ids != 0, 1, 0)

        # # NOTE: I have to add to dimension in between for the attention mask
        # # because it will be used to calculatation the attention scores
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        encoder_outputs = self.bert(input_ids, attention_mask = attention_mask, return_dict=True)

        # embeddings = self.embeddings(input_ids=input_ids)

        # Encoder ouputs can return the embeddings in each layer, but here
        # I only interested in the last hidden state
        # encoder_outputs = self.encoder(embeddings, attention_mask=attention_mask, return_dict=True)

        # head output
        if self.dqn_head:

            # NOTE: Instead of choosing the CLS token I will a random MASK token
            # input shape (batch_size, hidden_size, word_length)
            # Along the input shape check the mask token
            # If the mask token is present in the input_ids
            # I will use the mask token as the input to the head
            # If not I will use the CLS token
            # if self.tokenizer.mask_token_id in input_ids:
            #     mask_index = self._get_random_masked_token_mask(input_ids)
            #     output = self.head(encoder_outputs.last_hidden_state[mask_index], tried_letter)
            # else:
            output = self.head(encoder_outputs.pooler_output, tried_letter)

            # output = self.head(encoder_outputs.last_hidden_state[:, 0, :], tried_letter)
        else:
            output = self.head(encoder_outputs.last_hidden_state, tried_letter)

        # Mask the impossible actions
        # if self.training and self.dqn_head:
        #     if len(tried_letter.shape) == 1:
        #         tried_letter = tried_letter.unsqueeze(0)
        #     output[tried_letter == 1] -= 5 # Reduce the predicte q-value

        if not self.training and self.dqn_head:
        # if self.dqn_head:
            if len(tried_letter.shape) == 1:
                tried_letter = tried_letter.unsqueeze(0)
            output[tried_letter == 1] = -50 # Set a very low Q-value to don choose it

        # If I only processing one input I can return the output with the batch dimension
        if input_ids.shape[0] == 1:
            return output.squeeze(0)

        # return encoder_outputs.last_hidden_state, output
        return output
    
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        """
        LoRA linear layer.
        :param in_features: Input size of the linear layer.
        :param out_features: Output size of the linear layer.
        :param r: Rank of the low-rank adaptation.
        :param alpha: Scaling factor for the low-rank matrices.
        """
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.zeros_(self.weight)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (
            torch.nn.functional.linear(x, self.weight, self.bias)
            + self.scaling * torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.lora_A), self.lora_B
            )
        )

def apply_lora(bert_model, last_layers = 1, r=4, alpha=16):
    for i in range(-last_layers, 0):  # Replace the last two layers
        layer = bert_model.encoder.layer[i]
        
        # Replace attention linear layers with LoRA versions
        layer.attention.self.query = LoRALinear(
            in_features=layer.attention.self.query.in_features,
            out_features=layer.attention.self.query.out_features,
            r=r,
            alpha=alpha
        )
        layer.attention.self.key = LoRALinear(
            in_features=layer.attention.self.key.in_features,
            out_features=layer.attention.self.key.out_features,
            r=r,
            alpha=alpha
        )
        layer.attention.self.value = LoRALinear(
            in_features=layer.attention.self.value.in_features,
            out_features=layer.attention.self.value.out_features,
            r=r,
            alpha=alpha
        )
        
        # Replace intermediate and output dense layers with LoRA versions
        # layer.intermediate.dense = LoRALinear(
        #     in_features=layer.intermediate.dense.in_features,
        #     out_features=layer.intermediate.dense.out_features,
        #     r=r,
        #     alpha=alpha
        # )
        # layer.output.dense = LoRALinear(
        #     in_features=layer.output.dense.in_features,
        #     out_features=layer.output.dense.out_features,
        #     r=r,
        #     alpha=alpha
        # )

    return bert_model

def freezing_layers_and_LoRA(custom_bert):
    # Freeze all parameters
    for param in custom_bert.parameters():
        param.requires_grad = False

    # Unfreeze the last layer of the encoder
    # for param in custom_bert.encoder.layer[-1].parameters():
    #     param.requires_grad = True

    # Unfreeze the head
    for param in custom_bert.head.parameters():
        param.requires_grad = True

    # Apply LoRA to the last two layers
    custom_bert = apply_lora(custom_bert, r=4, alpha=16)


    # Count trainable parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = count_trainable_parameters(custom_bert)
    print(f"Number of trainable parameters: {trainable_params}")

    return custom_bert