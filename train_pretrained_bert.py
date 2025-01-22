from utils_model import CustomBERT
from utils_dataset import CharTokenizer, WordDataset
from trainer import BERTTrainer
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import wandb
import datetime
import os

from utils import save_checkpoint

# Load config.yaml
with open("config_bert.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set the name with the data of the experiment
current_date = datetime.datetime.now()
date_str = current_date.strftime("%Y_%m_%d-%H_%M_%S")  # Includes date and time
run_name = f"{config['run_name']}_{date_str}"

# Initialize W&B run with config
wandb.init(
    name=run_name,
    project=config["project_name"], 
    group=config["group_name"], 
    mode=config["mode"], 
    config=config  # Pass the entire config for hyperparameter tracking
)

# Print config
print("Config:")
for key, value in config.items():
    print(f"{key}: {value}")

# Access hyperparameters
cfg = wandb.config

# Bert config
hidden_size = cfg.hidden_size
num_hidden_layers = cfg.num_hidden_layers
num_attention_heads = cfg.num_attention_heads
max_position_embeddings = cfg.max_position_embeddings
intermediate_size = cfg.intermediate_size
max_word_length = cfg.max_word_length

# Training config
learning_rate = cfg.learning_rate
train_epochs = cfg.train_epochs
batch_size = cfg.batch_size
warmup_steps = cfg.warmup_steps
enable_schedule = cfg.enable_schedule
min_lr = cfg.min_lr

# Log config
save_model = cfg.save_model
save_model_freq = cfg.save_model_freq
enable_evaluation = cfg.enable_evaluation

# Load prev model

# Initialize Components
tokenizer = CharTokenizer()
vocab_size = len(tokenizer.vocab)
model = CustomBERT(
    vocab_size=vocab_size, # 4 special tokens
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    max_position_embeddings=max_position_embeddings,
    intermediate_size=intermediate_size,
)



def load_data(file_path):
    words = []
    with open(file_path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words

# Load the dataset in a list from a .txt file
train_data = load_data('data/filtered_words.txt')
test1_data = load_data('data/practice_words.txt')
test2_data = load_data('data/test_words_nltk.txt')

train_data = WordDataset(train_data, tokenizer, max_word_length)
test1_data = WordDataset(test1_data, tokenizer, max_word_length)
test2_data = WordDataset(test2_data, tokenizer, max_word_length)

train_loader = DataLoader(train_data,
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=4,
                          prefetch_factor=8,
                          pin_memory=True)
                        #   prefetch_factor=4)

test1_loader = DataLoader(test1_data,
                            batch_size=batch_size, 
                            pin_memory=True)

test2_loader = DataLoader(test2_data,
                            batch_size=batch_size,
                            num_workers=4,
                            prefetch_factor=8, 
                            pin_memory=True)

bert_trainer = BERTTrainer(model,
                            train_dataloader=train_loader,
                            test_dataloader=test1_loader,
                            lr= learning_rate,
                            warmup_steps=warmup_steps,
                            enable_schedule=enable_schedule,
                            min_lr=min_lr)

# Load the model
if cfg.load_model:
    print(f"Loading checkpoint: {cfg.load_model_path}")
    checkpoint = torch.load(cfg.load_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    bert_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if bert_trainer.optim_schedule and checkpoint['scheduler_state_dict']:
        for param_group, saved_group in zip(bert_trainer.optim_schedule._optimizer.param_groups, checkpoint['scheduler_state_dict']):
            param_group.update(saved_group)
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, train_epochs):
    train_stats = bert_trainer.train(epoch)
    wandb.log(train_stats, step=epoch)

    if enable_evaluation:
        test_stats1 = bert_trainer.test(epoch,
                                        data_loader=test1_loader,
                                        group_name="test1")
        wandb.log(test_stats1, step=epoch)

        test_stats2 = bert_trainer.test(epoch, 
                                        data_loader=test2_loader,
                                        group_name="test2")
        wandb.log(test_stats2, step=epoch)

    if save_model and (epoch+1) % save_model_freq == 0:
        print(f"Saving checkpoint at epoch {epoch}")
        # Create the directory
        path = f"models/bert/model_{date_str}"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/checkpoint_epoch_{epoch}.pth"
        save_checkpoint(epoch, model, bert_trainer.optimizer, bert_trainer.optim_schedule, save_path)


# Save the model
if save_model:
    print("Saving final checkpoint")
    path = f"models/bert/model_{date_str}"
    os.makedirs(path, exist_ok=True)
    save_path = f"{path}/checkpoint_final.pth"
    save_checkpoint(train_epochs, model, bert_trainer.optimizer, bert_trainer.optim_schedule, save_path)

wandb.finish()