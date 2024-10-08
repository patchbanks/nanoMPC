# nanoMPC

This repo contains a modified version of nanoGPT for fast and efficient generative MIDI drums. The nanoMPC model employs a simple technique of training a small transformer model with big data, leveraging extensive data augmentation to achieve generalization while maintaining a lightweight architecture. This approach ensures the model is not only powerful and coherent but also efficient for personalized training.

![Validation Loss](https://github.com/patchbanks/nanoMPC/blob/main/assets/nanompc_2M_val_loss.png?raw=true)

## Performance and Model Upgrades

The model is capable of learning various styles of drumming and beat-making (finger drumming) with intricate detail, including humanized velocities, rhythm swings, and rudimental patterns. These results are largely made possible through a normalization process and the use of custom augmentation tools, ensuring the model is trained with sufficient examples of expressive drum notations. Additionally, the positional encodings and attention mechanism were slightly modified to improve accuracy.

## Inference

Use `colab_run.ipynb` to generate MIDI files with the provided checkpoint file. This model was trained on a large dataset of 4-bar drum notations in the styles of Lo-Fi, Boom Bap, Throwback and other subgenres of Hip Hop.

For easy inference: [![nanoMPC](https://img.shields.io/badge/View%20nanoMPC%20on-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/patchbanks/nanoMPC)



## Contributing

We welcome data contributions for research and future development of our drum models. Data contributors may be given access to our augmentation tools to generate large-scale datasets. Additionally, musicians can receive free MIDI files for rating content. By providing human feedback, you will help us improve the model and enhance its capabilities.

***

To inquire about training a custom model, contributing data or rating content, please contact us.

_Our sincere appreciation to Andrej Karpathy for creating nanoGPT and his continuous support for the AI research community._
