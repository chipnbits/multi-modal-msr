# Third-party notices

This repository is MIT-licensed (see `LICENSE`). The files below contain
code vendored or adapted from third-party MIT-licensed projects; their
original copyright notices apply to those portions.

## `src/magsr/models/nn.py` and `src/magsr/models/unet.py`

Vendored from `torchcfm.models.unet` (conditional-flow-matching,
MIT License, Copyright (c) 2023 Alexander Tong):
https://github.com/atong01/conditional-flow-matching

which in turn vendors OpenAI guided-diffusion (MIT License,
Copyright (c) 2021 OpenAI):
https://github.com/openai/guided-diffusion

`AttentionPool2d` in `unet.py` is adapted from OpenAI CLIP (MIT License,
Copyright (c) 2021 OpenAI):
https://github.com/openai/CLIP

The MIT License text for the above:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
