import json
import argparse

import pandas as pd
import numpy as np
import torch
from torch import nn, cuda

from dataset import TrainDataset, DATA_ROOT, build_dataframe_from_folder
from transforms import get_test_transform
from main import get_model, make_loader, CACHE_DIR, ImageClassificationBot

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def get_class_idx_to_class_name_mapping(folder_to_idx):
    df_class_name = pd.read_csv(
        DATA_ROOT / "imagenet_class_map.txt", sep=" ", header=None)
    folder_to_name = {x: y for x, y in zip(df_class_name[0], df_class_name[2])}
    idx_to_name = {
        idx: folder_to_name[folder]
        for folder, idx in folder_to_idx.items()
    }
    return idx_to_name


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch-size', type=int, default=32)
    arg('--workers', type=int, default=4)
    arg('--arch', type=str, default='seresnext50')
    arg('--amp', type=str, default='')
    arg('--size', type=int, default=192)
    arg('--debug', action='store_true')
    arg('--model-path', type=str, default='')
    args = parser.parse_args()

    train_dir = DATA_ROOT / 'train'
    valid_dir = DATA_ROOT / 'val'

    use_cuda = cuda.is_available()
    model = get_model(args.arch)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    if use_cuda:
        model = model.cuda()
    if args.amp:
        if not APEX_AVAILABLE:
            raise ValueError("Apex is not installed!")
        model = amp.initialize(
            model, opt_level=args.amp
        )

    # The first line is to make sure we have the same class_map as in training
    _, class_map = build_dataframe_from_folder(train_dir)
    df_valid = build_dataframe_from_folder(valid_dir, class_map)
    idx_to_name = get_class_idx_to_class_name_mapping(class_map)
    # Export the mapping for later use
    with open(CACHE_DIR / "id_to_name_map.json", "w") as fout:
        json.dump(idx_to_name, fout)

    test_transform = get_test_transform(int(args.size*1.25), args.size)

    valid_loader = make_loader(
        args, TrainDataset, df_valid, test_transform, shuffle=False)

    print(f'{len(valid_loader.dataset):,} in valid')

    bot = ImageClassificationBot(
        model=model, train_loader=None,
        valid_loader=None, clip_grad=0,
        optimizer=None, echo=True,
        criterion=None,
        callbacks=[],
        pbar=True, use_tensorboard=False,
        use_amp=(args.amp != '')
    )
    logits, truths = bot.predict(valid_loader, return_y=True)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=1)
    print(
        f"Validation accuracy: {np.mean(preds.numpy() == truths.numpy()) * 100:.2f}%"
    )
    df_out = pd.DataFrame({
        "truth": truths.numpy(),
        "max_prob": np.max(probs.numpy(), axis=1),
        "truth_prob": torch.gather(probs, 1, truths[:, None]).numpy()[:, 0],
        "pred": preds,
        "path": [
            valid_loader.dataset._df.iloc[i].image_path
            for i in range(len(valid_loader.dataset))
        ]
    })
    df_out.to_csv(CACHE_DIR / "valid_preds.csv", index=False)


if __name__ == '__main__':
    main()
