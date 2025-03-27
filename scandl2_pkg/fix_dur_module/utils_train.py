



import torch 
import torch.nn as nn
import numpy as np 
import transformers

from typing import Optional


class EarlyStopping: 
    def __init__(
        self,
        patience: int,
        path: str,
        delta: Optional[int] = 0,
    ):
        self.patience = patience
        self.delta = delta 
        self.best_score = None 
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf 
        self.path = path 

    def __call__(
        self,
        val_loss,
        model,
    ):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score 
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
    def save_checkpoint(
        self,
        val_loss, 
        model,
    ):
        """ Saves model when validation loss decreases. """
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss 



def train(
    model, 
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.MSELoss,
    optimizer: transformers.AdamW,
    early_stopping: EarlyStopping,
    scheduler: transformers.get_linear_schedule_with_warmup,
    device: torch.device,
    fix_dur_colname: str,
    output_attentions: Optional[bool] = None,
    use_attention_mask: Optional[bool] = None,
):
    """
    Train loop to train the Seq2Seq model.
    :param model: the model to train
    :param num_epochs: number of epochs to train
    :param train_loader: the training data loader
    :param val_loader: the validation data loader
    :param criterion: the loss function (MSE Loss)
    :param optimizer: the optimizer (AdamW)
    :param early_stopping: the early stopping object
    :param scheduler: the learning rate scheduler
    :param device: the device to train on
    :param fix_dur_colname: the name of the column containing the fixations durations
    """
    for epoch in range(num_epochs):

        model.train()

        for batch_idx, train_batch in enumerate(train_loader):

            optimizer.zero_grad()

            sp_embeddings = train_batch['sp_embeddings'].to(device)
            attention_mask = train_batch['attention_masks'].to(device)
            fix_durs = train_batch[fix_dur_colname].to(device)

            # forward pass
            if use_attention_mask:

                if output_attentions:

                    out, _ = model(
                        sp_embeddings=sp_embeddings,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                    )
                else:

                    out = model(
                        sp_embeddings=sp_embeddings,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                    )
            else:

                if output_attentions:

                    out, _ = model(
                        sp_embeddings=sp_embeddings,
                        output_attentions=output_attentions,
                    )
                else:
                    out = model(
                        sp_embeddings=sp_embeddings,
                        output_attentions=output_attentions,
                    )

            #train_loss = criterion(out, fix_durs)
            # mask the padding in the loss computation
            loss_mask = (fix_durs != 0).float()
            #train_loss = criterion(out * loss_mask, fix_durs * loss_mask)
            train_loss = criterion(out, fix_durs)

            train_loss.backward()
            optimizer.step()
            scheduler.step()

            print(f'\t  epoch {epoch+1}, batch {batch_idx+1}, loss: {train_loss.item():.4f}')                        
        
        # validation
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            
            for val_batch in val_loader:

                sp_embeddings = val_batch['sp_embeddings'].to(device)
                attention_mask = val_batch['attention_masks'].to(device)
                fix_durs = val_batch['fix_durs'].to(device)

                if use_attention_mask:

                    if output_attentions:
                        out, attentions = model(
                            sp_embeddings=sp_embeddings,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                        )
                    else:
                        out = model(
                            sp_embeddings=sp_embeddings,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                        )
                else:

                    # forward pass
                    if output_attentions:
                        out, attentions = model(
                            sp_embeddings=sp_embeddings,
                            output_attentions=output_attentions,
                        )
                    
                    else:
                        out = model(
                            sp_embeddings=sp_embeddings,
                            output_attentions=output_attentions,
                        )

                val_loss_mask = (fix_durs != 0).float()
                #val_loss += criterion(out * val_loss_mask, fix_durs * val_loss_mask).item()
                val_loss += criterion(out, fix_durs).item()
        
        # average the losses
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # check for early stopping 
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break