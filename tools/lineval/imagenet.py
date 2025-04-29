import os
from argparse import ArgumentParser
from tqdm.auto import tqdm

import torch
from torch import optim
from torch.optim import lr_scheduler

from mdistiller.dataset.imagenet import get_imagenet_dataloaders
from mdistiller.models._base import ModelBase

from tools.lineval.utils import init_parser, load_from_checkpoint


if __name__ == '__main__':
    parser = ArgumentParser('lineval.imagenet')
    init_parser(parser)
    args = parser.parse_args()
    
    DEVICE = args.device
    EPOCHS = args.epochs
    
    train_loader, test_loader, _ = get_imagenet_dataloaders(
        args.batch_size, args.test_batch_size,
        args.num_workers, use_ddp=False,
    )
    model, _ = load_from_checkpoint(args.expname, tag=args.tag, expected_arch='transformer')
    model: ModelBase = model.cuda(DEVICE)
    head = torch.nn.Linear(model.get_head().in_features, 1000).cuda(DEVICE)
    optimizer = optim.SGD(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS*len(train_loader),
        eta_min=1.0E-8,
    )
    
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True) as bar:
            total_loss, correct, total = 0, 0, 0
            for input, target, _ in bar:
                with torch.no_grad():
                    x = model.forward_stem(input.cuda(DEVICE))
                    x = model.get_layers().forward(x)
                    x = model.forward_pool(x)
                logit = head.forward(x)
                pred = logit.argmax(dim=1).clone().detach().cpu()
                
                loss = torch.nn.functional.cross_entropy(logit, target.cuda(DEVICE))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                batch_size = input.size(0)
                total_loss += loss.clone().detach().cpu().item() * batch_size
                correct += (pred == target).sum().item()
                total += batch_size
                bar.set_postfix(dict(
                    top1=correct/total*100,
                    loss=total_loss/total,
                    lr=optimizer.param_groups[0]['lr'],
                ))
                break

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True) as bar, torch.no_grad():
            total_loss, correct, total = 0, 0, 0
            for input, target in bar:
                x = model.forward_stem(input.cuda(DEVICE))
                x = model.get_layers().forward(x)
                x = model.forward_pool(x)
                logit = head.forward(x)
                pred = logit.argmax(dim=1).cpu()
                loss = torch.nn.functional.cross_entropy(logit, target.cuda(DEVICE))
                
                batch_size = input.size(0)
                total_loss += loss.cpu().item() * batch_size
                correct += (pred == target).sum().item()
                total += batch_size
                bar.set_postfix(dict(
                    top1=correct/total*100,
                    loss=total_loss/total,
                ))
