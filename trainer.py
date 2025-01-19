import torch
import math
import tqdm
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class BERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr= 1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000
        ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")

        self.model = model.to(self.device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optimizer, self.model.config.hidden_size, n_warmup_steps=warmup_steps
            )
        
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch, data_loader=None, group_name=None):
        if data_loader is None:
            data_loader = self.test_data
        return self.iteration(epoch, data_loader, train=False, group_name=group_name)

    def iteration(self, epoch, data_loader, train=True, group_name = None):
        
        avg_loss = 0.0
        total_correct = 0  # To track correct predictions
        total_masked = 0   # To track total masked tokens

        if train:
            mode = "train"
            self.model.train()
        else:
            mode = "test"
            self.model.eval()

        if group_name is None:
            group_name = mode
            
        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (group_name, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:

            # batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # forward the model
            mask_lm_output = self.model.forward(data["bert_input"], data["attention_mask"])

            # NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            
            # NOTE: the mask_lm_output will return -log probability values,
            # then the criterion will only average the values of the masked tokens
            loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                # self.optimizer.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                # self.optimizer.step()

            # Update average loss
            avg_loss += loss.item()

            # Calculate predictions and accuracy
            predictions = torch.argmax(mask_lm_output, dim=-1)  # Shape: (batch_size, seq_len)
            correct = (predictions == data["bert_label"]) & (data["bert_label"] != 0)  # Exclude padding
            total_correct += correct.sum().item()
            total_masked += (data["bert_label"] != 0).sum().item()  # Exclude padding tokens

            # Calculate perplexity
            # NOTE: perplexity is not well defined for masked language models like BERT (see summary of the models).
            perplexity = math.exp(avg_loss / (i + 1))

            # Calculate masked token accuracy
            accuracy = total_correct / total_masked if total_masked > 0 else 0

        
        stats = {
            # f"{group_name}/epoch": epoch,
            f"{group_name}/avg_loss": round(avg_loss / (i + 1), 3),
            f"{group_name}/loss": round(loss.item(), 3),
            f"{group_name}/perplexity": round(perplexity, 3),
            f"{group_name}/accuracy": round(accuracy, 3)
        }

        return stats