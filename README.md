# PyTorch Helper Bot

[WIP] a high-level PyTorch helper package

This project is intended for my personal use. Backward compatibility will not be guaranteed. Important releases will be tagged.

## Motivation

[_fast.ai_](https://github.com/fastai/fastai) is great, and I recommend it for all deep learning beginners. But since it's beginner-friendly, a lot of more sophisticated stuffs are abstracted heavily and hidden from users. Reading the source code is often required before you can tweak the underlying algorithms. The advent of `doc` function greatly speeds up the process by quickly directing the user to the source code and documentation.

However, _fast.ai_ has become stronger and bigger. Not everyone has time to keep up with its codebase. Hence the creation of this project. I built a relatively thin layer of abstraction upon PyTorch from scratch, with a lot of ideas and code borrowed from various sources (mainly _fast.ai_). Only features that are relevant to my use cases are added.

Another similar project is [pytorch/ignite](https://github.com/pytorch/ignite).

## Tips

- Set the environment variable `SEED`, and we will set the random seed of Python, Numpy and PyTorch for you.
- Set the environment variable `DETERMINISTIC` to make CuDNN operations deterministic.
- Set `pbar=False` when initializing a bot to disable progress bar during evaluation.

## Examples

There are almost no unit tests yet. The following example(s) are somewhat functional tests.

- [Imagenette Image Classification](examples/imagenette/)
