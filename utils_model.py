import torch
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers import BertConfig, BertModel, BertTokenizer
from torch.nn import Module

# Head using embedding layer
# If I use the 300 Glove embeddings I can use the embedding layer to predict the word
# class MLMHead(torch.nn.Module):
#     def __init__(self, embedding_layer):
#         """
#         :param embedding_layer: Embedding layer from the model
#         """
#         super().__init__()
#         # Use the embedding layer's weight matrix for the linear layer
#         self.linear = torch.nn.Linear(embedding_layer.word_embeddings.weight.size(1),
#                                        embedding_layer.word_embeddings.weight.size(0))
#         self.linear.weight = embedding_layer.word_embeddings.weight  # Share weights
#         self.softmax = torch.nn.LogSoftmax(dim=-1)

#     def forward(self, x):
#         return self.softmax(self.linear(x))

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
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, max_position_embeddings, intermediate_size):
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
        # self.pooler = BertPooler(config)

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