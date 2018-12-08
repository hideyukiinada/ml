What kind of painting do you think Picasso would produce if he were still alive today?
Or, wouldn't it so cool if you can see another art by Andy Warhol?
GAN, or Generative Adversarial Network has potential to bring these dreams come close to reality.
GAN was invented by Ian Goodfellow in 2014.  You can refer to his original paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) if you are interested.

In short, GAN consists of two software objects:
* Generator
* Discriminator

Generator, who is responsible for generating a real-looking fake image from noise.
Discriminator, who is responsible for telling the fake objects from real objects.
During a training process, you alternate training for Discriminator and Generator.  As you iterate through many cycles, Generator learns to produce more real-looking fake images.
Now GAN became very popular and there are many variants.  Most notably DCGAN and CYCLE GAN.  The latter can generate images of zebras from horse images!

In this article, I would like to dive a little deeper and explaine how I implemented GAN to generate digit (number) looking images from scratch.


