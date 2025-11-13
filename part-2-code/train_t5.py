import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
GEN_CONFIG = GenerationConfig(
    max_new_tokens=128,
    num_beams=4,
    early_stopping=True,
)

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=10,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=3,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

from pathlib import Path
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import T5TokenizerFast
 
def eval_epoch(args, model, dev_loader, gt_sql_pth, pred_sql_pth, gt_record_pth, pred_record_pth):
    model.eval()

    _tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    _bos = _tok.convert_tokens_to_ids("<extra_id_0>")

    crit = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    total_loss_sum = 0.0
    total_token_cnt = 0
    all_sql_predictions = []

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            enc_ids, enc_mask, dec_in, dec_tgt, _init_dec_in = batch
            enc_ids = enc_ids.to(DEVICE, non_blocking=True)
            enc_mask = enc_mask.to(DEVICE, non_blocking=True)
            dec_in = dec_in.to(DEVICE, non_blocking=True)
            dec_tgt = dec_tgt.to(DEVICE, non_blocking=True)

            
            out = model(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                decoder_input_ids=dec_in,
            )
            logits = out["logits"]  

            
            B, T, V = logits.size()
            flat_logits = logits.view(B * T, V)
            flat_targets = dec_tgt.view(B * T)

            per_tok = crit(flat_logits, flat_targets)              
            non_pad = (flat_targets != PAD_IDX)                   
            masked = per_tok[non_pad]

            total_loss_sum += masked.sum().item()
            total_token_cnt += non_pad.long().sum().item()

            
            gen = model.generate(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                decoder_start_token_id=_bos,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                length_penalty=0.8,
            )
            all_sql_predictions.extend(_tok.batch_decode(gen, skip_special_tokens=True))

    avg_loss = (total_loss_sum / total_token_cnt) if total_token_cnt else 0.0

   
    Path(pred_sql_pth).parent.mkdir(parents=True, exist_ok=True)
    Path(pred_record_pth).parent.mkdir(parents=True, exist_ok=True)

    
    save_queries_and_records(all_sql_predictions, pred_sql_pth, pred_record_pth)

    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, pred_sql_pth, gt_record_pth, pred_record_pth
    )
    err_rate = (sum(1 for m in error_msgs if m != "") / len(error_msgs)) if error_msgs else 0.0

    return avg_loss, record_f1, record_em, sql_em, err_rate


def test_inference(args, model, test_loader, pred_sql_pth, pred_record_pth):

    model.eval()

    _tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    _bos = _tok.convert_tokens_to_ids("<extra_id_0>")

    preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            enc_ids, enc_mask, _init_dec_in = batch
            enc_ids = enc_ids.to(DEVICE, non_blocking=True)
            enc_mask = enc_mask.to(DEVICE, non_blocking=True)

            gen = model.generate(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                decoder_start_token_id=_bos,
                max_length=256,
                num_beams=5,
                early_stopping=True,
            )
            preds.extend(_tok.batch_decode(gen, skip_special_tokens=True))

    Path(pred_sql_pth).parent.mkdir(parents=True, exist_ok=True)
    Path(pred_record_pth).parent.mkdir(parents=True, exist_ok=True)

    save_queries_and_records(preds, pred_sql_pth, pred_record_pth)

    with open(pred_record_pth, "rb") as fh:
        _records, err_msgs = pickle.load(fh)

    err_rate = (sum(1 for m in err_msgs if m != "") / len(err_msgs)) if err_msgs else 0.0

    print(f"[Test] SQL execution errors: {err_rate * 100:.2f}%")
    print(f"[Test] Total generated queries: {len(preds)}")
    print(f"[Test] Files saved:\n  - SQL:  {pred_sql_pth}\n  - PKL:  {pred_record_pth}")


def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()

