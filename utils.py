
# Print the config in a more readable format
# print("Config:")
# for key, value in config.items():
#     print(f"{key}: {value}")
#
import torch

class AttrDict:
    """
    A class to convert a dictionary to an object where keys can be accessed as attributes.
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):  # Recursively handle sub-dictionaries
                value = AttrDict(value)
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return str(self.__dict__)
    
def save_checkpoint(epoch, model, optimizer, scheduler, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler._optimizer.param_groups if scheduler else None
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch}: {save_path}")

def transfer_weights(source_model, custom_bert_model, layer_count=2):
    # Extract state_dict from CustomBERT
    custom_bert_state_dict = custom_bert_model.state_dict()

    # Map embeddings
    custom_bert_state_dict['embeddings.word_embeddings.weight'] = source_model['embeddings.word_embeddings.weight']
    custom_bert_state_dict['embeddings.position_embeddings.weight'] = source_model['embeddings.position_embeddings.weight']
    custom_bert_state_dict['embeddings.token_type_embeddings.weight'] = source_model['embeddings.token_type_embeddings.weight']
    custom_bert_state_dict['embeddings.LayerNorm.weight'] = source_model['embeddings.LayerNorm.weight']
    custom_bert_state_dict['embeddings.LayerNorm.bias'] = source_model['embeddings.LayerNorm.bias']

    # Map encoder layers
    for i in range(layer_count):  # Assuming 2 transformer layers
        for key in ['attention.self.query.weight', 'attention.self.query.bias',
                    'attention.self.key.weight', 'attention.self.key.bias',
                    'attention.self.value.weight', 'attention.self.value.bias',
                    'attention.output.dense.weight', 'attention.output.dense.bias',
                    'attention.output.LayerNorm.weight', 'attention.output.LayerNorm.bias',
                    'intermediate.dense.weight', 'intermediate.dense.bias',
                    'output.dense.weight', 'output.dense.bias',
                    'output.LayerNorm.weight', 'output.LayerNorm.bias']:
            custom_bert_state_dict[f'encoder.layer.{i}.{key}'] = source_model[f'encoder.layer.{i}.{key}']

    # Load the updated state_dict into CustomBERT
    custom_bert_model.load_state_dict(custom_bert_state_dict)

    return custom_bert_model