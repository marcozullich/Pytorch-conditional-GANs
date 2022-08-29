# Conditional Deep Convolutional Generative Adversarial Network 

Conditional Generation of MNIST images using conditional DC-GAN in PyTorch.

Based on the following papers:
* [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

Implementation inspired by the PyTorch examples [implementation of DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).

Updated by [Marco Zullich](zullich.it).

## Sample Results
Example of sampling results shown below. Each row is conditioned on a different digit label:
![Example of sampling results](sample_outputs/samples1.png)

## Training

Run `conditional_dcgan.py` with the following args (remove `--cuda` if you don't have a CUDA-capable GPU).
Append `--clear_save_dir` to remove clutter.

```
python conditional_dcgan.py --cuda --save_dir=models --samples_dir=samples --epochs=25
```

## Inference

Run

```
generate.py --weights <path_to_generator_params>
```

You may download the pre-trained parameters from [here](https://drive.google.com/file/d/1yJ3Bq-DHcO4bqi-rGVD_r79M-asNeFpP/view?usp=sharing).

Defaults are 5 images from random classes. Use `--num_images` for a different amount. Control the classes by setting the parameter `--categories`. You may specify a single number for generating images from the same category, or specify a sequence of numbers of the same length as `--num_images`.
The images will be saved in `generated/`.

## Questions and comments:

Feel free to reach to me at `malzantot [at] ucla [dot] edu` for any questions or comments.
