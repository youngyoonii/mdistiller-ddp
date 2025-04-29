from argparse import ArgumentParser
from tqdm.auto import tqdm

import einops

import torch
from torch import optim
from torch.optim import lr_scheduler

from mdistiller.dataset.nyud_v2 import get_nyud_dataloaders
from mdistiller.models._base import ModelBase

from tools.lineval.utils import init_parser, load_from_checkpoint


if __name__ == '__main__':
    parser = ArgumentParser('lineval.imagenet')
    init_parser(parser, defaults=dict(epochs=5000))
    args = parser.parse_args()
    
    DEVICE = args.device
    EPOCHS = args.epochs
    
    train_loader, test_loader, _ = get_nyud_dataloaders(
        args.batch_size, args.test_batch_size,
        args.num_workers, use_ddp=False,
    )
    model, _ = load_from_checkpoint(args.expname, tag=args.tag, expected_arch='transformer')
    model: ModelBase = model.cuda(DEVICE)
    head = torch.nn.Linear(model.embed_dim, 768).cuda(DEVICE)
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
    
    def patches_to_depth(x: torch.Tensor):
        _, num_patches, embed_dim = x.shape
        patch_rows = int(num_patches**0.5)
        pixel_rows = int(embed_dim**0.5)
        return einops.rearrange(
            x, 'b (ph pw) (h w) -> b (ph h) (pw w)',
            ph=patch_rows, pw=patch_rows,
            h=pixel_rows, w=pixel_rows,
        )
    
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True) as bar:
            total_loss, correct, total = 0, 0, 0
            for input, target in bar:
                with torch.no_grad():
                    x = model.forward_stem(input.cuda(DEVICE))
                    x = model.get_layers().forward(x)
                pred_patches = head.forward(x)
                pred = patches_to_depth(pred_patches)
                
                loss = torch.nn.functional.mse_loss(pred, target.to(DEVICE))
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

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True) as bar, torch.no_grad():
            total_loss, correct, total = 0, 0, 0
            for input, target in bar:
                x = model.forward_stem(input.cuda(DEVICE))
                x = model.get_layers().forward(x)
                pred_patches = head.forward(x)
                pred = patches_to_depth(pred_patches)
                loss = torch.nn.functional.mse_loss(pred, target.to(DEVICE))
                
                batch_size = input.size(0)
                total_loss += loss.cpu().item() * batch_size
                correct += (pred == target).sum().item()
                total += batch_size
                bar.set_postfix(dict(
                    top1=correct/total*100,
                    loss=total_loss/total,
                ))
