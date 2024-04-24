# SimpleMambaclassifier

1. Here you will find full code of implementation of Mambacclassifier

![picturewithschema](https://github.com/teRarity/Mambacclassifier/assets/123636828/2febd6d8-f072-424a-94b3-7750918ccf6e)

## Abstract

Image classification is a fundamental task in computer vision, and Convolutional Neural Networks (CNNs) have been widely used for this purpose. However, CNNs often struggle with capturing long-range dependencies and have quadratic computational complexity. Inspired by the recent success of the state space model (SSM) represented by Mamba, I proposed a simplified version named SimpleMambaclassifier for classifying normal images like CIFAR-10.

Vision Mamba introduces a novel Conv-SSM module that combines the local feature extraction capabilities of convolutional layers with the ability of SSM to model long-range interactions efficiently. By incorporating SSM into the convolutional architecture, SimpleMambaclassifier aims to capture both local and global dependencies in images.

To demonstrate the effectiveness of SimpleMambaclassifier, I conducted extensive experiments using the CIFAR-10 dataset, which consists of normal images from various classes. I compared the performance of Vision Mamba with traditional CNNs and found that it achieves superior results while maintaining linear computational complexity

![image](https://github.com/teRarity/Mambacclassifier/assets/123636828/170deee5-e144-47d3-b49d-3c5c5c3e8015)

