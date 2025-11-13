import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_MODEL_NAME = "google-t5/t5-small"
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = T5TokenizerFast.from_pretrained(BASE_MODEL_NAME)
    return _TOKENIZER

def setup_wandb(args):
    if wandb.run is not None:
        return
    wandb.init(project="hw4-part2-t5", name=args.experiment_name, config=vars(args))

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        # Load pretrained T5-small model for finetuning
        print("Loading pretrained T5-small model for finetuning...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        model.config.dropout_rate = 0.1
        model.config.attention_dropout_rate = 0.1
    else:
        # Initialize T5-small from scratch (random weights)
        print("Initializing T5-small model from scratch...")
        config = T5Config.from_pretrained('google-t5/t5-small',  dropout_rate=0.1,
            attention_dropout_rate=0.1)
        model = T5ForConditionalGeneration(config)
    
    num_layers_to_freeze = 1
    for name, param in model.named_parameters():
        if name.startswith("encoder.block.") and int(name.split(".")[2]) < num_layers_to_freeze:
            param.requires_grad = False
    
    # Print how many layers are frozen
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    model = model.to(DEVICE)
    
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    tag = "best" if best else "last"
    save_dir = os.path.join(checkpoint_dir, tag)
    mkdir(save_dir)
    model.save_pretrained(save_dir)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    tag = "best" if best else "last"
    ckpt_dir = os.path.join(args.checkpoint_dir, tag)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found")
    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir)
    tokenizer = _get_tokenizer()
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    model.to(DEVICE)
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
