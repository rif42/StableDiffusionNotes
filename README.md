---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
---

# Stable Diffusion

Stable diffusion is a generative AI tech that creates and manipulates images from prompts or other images. The model uses labeled images as training data and outputs images or modification of images.  
To use stable diffusion model, we input a text prompt, then the model outputs an image. This means Stable Diffusion has multiple models inside it.  
Here's how it works :

## The Image Layer   
The main function of the image layer is to extract all the possible information inside an image, and encode it into a digestible format.  
### Convolutional Network  
Convolutional network is a type of neural network that has a 3x3 matrix (called kernel) moving to scan every area of an image to extract its features such as lines, shadows, highlights, color, saturation, etc. [[1]](https://ieeexplore.ieee.org/abstract/document/8308186) [[2]](https://arxiv.org/abs/1511.08458)  

### Semantic Segmentation with U-Net  
Semantic segmentation is a task to segment image into multiple classes and tell them what they are. This task is achieved by using convolutional network with specialized kernels. [[3]](https://ieeexplore.ieee.org/abstract/document/8354267)  

![image segmentation](./assets/segmentation.png)  

One of the most efficient semantic segmentation algorithm is U-Net. [[4]](https://ieeexplore.ieee.org/abstract/document/9446143)   

![unet arch](./assets/unet.png)  

The U-Net is trained using raw images and the images' perfect black-and-white mask as the ground truth (or target).  
If normal convolutional models scans images in 2 dimensions (width and length), The U-Net scans images in 3 dimensions (width, length, and channels). An image consists of 3 channels, Red, Green, and Blue. Each channel can make their own monochromic image.   

![rgb](./assets/rgb.png)  

At the same time, U-Net also compresses the size of the image after each cycle, increasing field of view, while keeping the same kernel size, making it able to extract more and more features.  

We are now at the bottom of the U curve, the model has learned all the features of the image. Now its time to scale the image back up to original size.  

The U-net up-samples the image based on the image received before the image is getting compressed (hence the copy and crop grey arrow)  

### U-Net as Image Denoiser  
Because U-Net does such a good job at semantic segmentation, people are experimenting with using U-Net as an image denoiser. Noise is unwanted pixel value deviations in the image that can alter the overal perception of the image. Image denoiser's primary job is to remove noise. [[5]](https://openaccess.thecvf.com/content/ICCV2021W/NeurArch/html/Jia_DDUNet_Dense_Dense_U-Net_With_Applications_in_Image_Denoising_ICCVW_2021_paper.html)  

If we generate a completely random pixels which constitutes a pure noise image, then we add a (perfect) image of fish on top of it, the result is a noised image of a fish.  
That means, if we have a noised image of a fish, then subtract the pure noise image out of it, then we have a perfect image of a fish.   

![noised fish](./assets/noise.png)    

In this image denoising model, the input is a noised image of a fish, and the ground truth is the pure noise image. However, we want to train this model in small steps. Each step, the noise going to be removed ever so slightly. Over many many steps (lets say 10, or even 1000), we will get the clean, perfect denoised image.  

However, the model does not know how noisy the input image is. To do that we will use positional encoding or embedding. Positional encoding is converting discrete variables (image sequence numbers) into continuous vectors so its easily processed by the model.  
This embedded is added into every step of the red arrow (maxpool/compression) and green arrow (upsample) as shown in the U-Net architecture.  

### Auto-Encoder  
Auto-encoder refers to a neural network that encodes data, compresses it into a smaller representation which is called latent space, then decode the latent space into the original data form.  
What is encoding, latent space, and decoding? Take a look at this [website](https://n8python.github.io/mnistLatentSpace/)   
The MNIST dataset image resolution is 28x28 is represented by 784 parameters times the amount of images in it.  
Then all the images are encoded into latent space of just 2 parameters (x and y coordinates) so we can map it on a 2d plane.  
Once we select the coordinates, the latent space parameters gets decoded into 28x28 image, resembling the original dataset.   

![flow of stable diffusion](./assets/diffusionflow.png)    

In Stable Diffusion, the trained images gets encoded (compressed) into smaller latent space.  
Then we perform the noising/denoising training steps.  
Finally, we decode that modified image in the latent space into its original resolution.
Operating the image in latent space significantly decreasing processing times and increasing feature stability. This is what the **stable** means in Stable Diffusion. [[6]](https://deci.ai/deep-learning-glossary/stable-diffusion/)  

## The Word Layer  
The main function of the word layer is to extract all the possible context a sentence or word, and encode it into a digestible format.  

### Text Embedding  
We need a way to represent words into something a model can digest. The answer is embedding using word2vec. [Computerphile](https://www.youtube.com/watch?v=gQddtTdmG_8) made a great video about this.  
So we gather all the english words into one database, then we put an index to each word.  
Then we encode the index using positional encoding, converting an integer into a vector (just like embedding in U-Net).  
However, we will create a copy of the database, but this time with randomized index, then encode it using positional encoding too.  
After that, we look for training data that contains perfect grammar from dictionary, internet articles, comments, captioned speeches, etc.  
Then we count how many times x word is positioned next to y words, then build a relationship model from it.  
For example, there are 2 word-pairings, "cute kitten" and "cute bottle". Because "cute kitten" appears much more frequently than "cute bottle", it means that the word "cute" is more likely to be followed with "kitten" as opposed to "bottle"  

You can check this demo of fully graphed multidimensional text embedding in this [Website](https://projector.tensorflow.org/)   

The coolest thing is, because the text embedding is made up of vectors, adding and subtracting vectors is very similar to adding and subtracting *meaning* in real life context.  

![vector operation on text embedding](./assets/vec.png)    

### Self Attention  
Self attention main function is to extract features from phrases using relationships between words depending on their text embeddings vectors. It's one of the most incredible breakthrough in machine learning field [[7]](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)    

![vector operation on text embedding](./assets/att.png)    

We take the vectors for each word, and turn it into a matrix.  
For example, in language translation, a model should translate the essence of a phrase, and that doesn't necessarily mean translating everything word-by-word. You can also watch [this amazing video by google](https://www.youtube.com/watch?v=fjJOgb-E41w)  

English : black cat ate the mouse  
French : le chat noir a mange la souris  
chat means cat, and noir means black. The french translation inverts the original word positioning, from black cat to chat noir (cat black).  
The self-attention model aims to account for this issue by prioritizing words that are more important than the rest of the words in the phrase.  
However, to achieve that, the model needs to know *where* the important word (or words) is.  
To do that, we create a positional encoding of each word, than sum it into the original vector of each word.  

```
Word Embeddings:
Word 1: [0.1, 0.2, 0.3, 0.4]
Word 2: [0.5, 0.6, 0.7, 0.8]
...

Positional Encodings:
Position 1: [0.9, 0.8, 0.7, 0.6]
Position 2: [0.1, 0.2, 0.3, 0.4]
...

Input to Self-Attention:
Word 1 + Position 1 = [1.0, 1.0, 1.0, 1.0]
Word 2 + Position 2 = [0.6, 0.8, 1.0, 1.2]
...
```
Now, we need to determine the important words in the phrase and its relationship with other words, such as negation, contradiction, emphasize, etc.  

![relationship between words](./assets/att2.png)    

To do this, we use the query, key and value calculation.  
The self-attention mechanism computes attention scores between the query vector of a word and the key vectors of all the words in the sequence. These attention scores are then used to weight the value vectors, resulting in an output vector for the query word. The output vector captures information from the parts of the sequence that are most relevant to the query word, and this process is performed for each word in the sequence.  

the query, key, and value vectors are used in self-attention to determine how much attention each word should pay to other words in the sequence, and the weighted values are aggregated to create meaningful representations for each word.   

Unfortunately, we cannot influence this calculation in any way. And if we can't influence, that means we can't train it.  
To solve this, we add a simple linear neural network as a parameter to the query, key and value before they are calculated. This parameter is called hidden layer  

![hiddenlayer](./assets/hiddenlayer.png)    

## Combining Image and Word Layer  

![combination between image and word layer](./assets/combination.png)   

We have convolutional encoder to encode image into an embedding.  
We have self attention to encode text into an embedding.  
That means perfectly labeled images should have very similar image embedding and text embedding.  
To combine these two together, we have to inject our self attention text encoder into our u-net convolutional encoder at every step of noise-denoise proccess.  

![combination between image and word layer](./assets/inject.png)   

However, when applied in this manner, we need to use cross-attention model instead of self-attention model.  
Self attention model operates on one set of data (text), while cross-attention model operates on two sets of data (image and text prompts)  

![cross attention layer](./assets/crossattention.png)   

The query is the image, while the key and value is the text prompt. This allows the model to generate all the objects and their relationship to other objects based on the text prompt.   

With this architecture, we're able to train the model by feeding it (perfectly) labeled images. The model will figure out the context of the label and the image. Finally, the model will generate new images using new text prompts based on the knowledge of the contexts and images it learned during the data training session.












































