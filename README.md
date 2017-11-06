# DeepLearning-Tutorials
  This is a deep learning tutorials for someone who want to learn more about AI, Thank you for following!
  
# Book List
## Chinese Book
  - 《机器学习实战》--Peter Harrington
    
  - 《机器学习》--周志华
    
  - 《统计和机器学习》--李航
    
  - 《神经网络与深度学习》--邱锡鹏（https://nndl.github.io/）
    
  - 《深度学习》--Ian GoodFellow, Yoshua Bengio et al(https://exacity.github.io/deeplearningbook-chinese/)
    
## English Book
  - 《Deep Learning》--Ian GoodFellow, Yoshua Bengio et al
  
  - 《Pattern Recognition and Machine Learning》--Christopher M. Bishop
     
# Courses List
 - Machine Learning--by Andrew Ng, Standford(https://www.coursera.org/learn/machine-learning)
  
 - Deep Learning -- by Andrew Ng, Standford(https://www.coursera.org/specializations/deep-learning)

 - CS231n: Convolutional Neural Networks for Visual Recognition--by Feifei Li(http://cs231n.stanford.edu/syllabus.html)
  
 - Stat212b：Topics Course on Deep Learning--by Joan Bruna(http://joanbruna.github.io/stat212b/)
  
 - CS224d: Deep Learning for Natural Language Processing--by Richard Socher(http://cs224d.stanford.edu/)
  
# Paper List
## Application
### Image Revolution and Convolution Networks

  **[0]** LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998d). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.(LeNet-5)
  
  **[1]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012. [pdf] (AlexNet, Deep Learning Breakthrough) :star::star::star::star::star:

  **[2]** Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014). [pdf] (VGGNet,Neural Networks become very deep!) :star::star::star:

  **[3]** Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [pdf] (GoogLeNet) :star::star::star:

  **[4]** He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015). [pdf] (ResNet,Very very deep networks, CVPR best paper) :star::star::star::star::star:
  
### Semantic Segmentation and Object Detection
  
  **[0]** 	Evan Shelhamer, Jonathan Long, Trevor Darrell:Fully Convolutional Networks for Semantic Segmentation. IEEE Trans. Pattern Anal. Mach. Intell. (2017)(FCN)
  
  **[1]** 	Ross B. Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik:Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. CVPR 2014(RCNN)
  
  **[2]** Ross Girshick, Redmond.Fast R-CNN: Fast Region-based Convolutional Networks for object detection. ICCV 2015(Fast RCNN):https://github.com/rbgirshick/fast-rcnn

  **[3]** 	Shaoqing Ren, Kaiming He, Ross B. Girshick, Jian Sun:Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NIPS 2015(Faster RCNN):https://github.com/ShaoqingRen/faster_rcnn
  
  **[4]** 	Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross B. Girshick:Mask R-CNN. CVPR (2017)(Mask RCNN)

### Natural Language Processing

### Speech Recognization

## Model
### RNN LSTM GRU etc.
   **[0]** Graves, Alex. "Generating sequences with recurrent neural networks." arXiv preprint arXiv:1308.0850 (2013). [pdf] (LSTM, very nice generating result, show the power of RNN) :star::star::star::star:

   **[1]** Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." arXiv preprint arXiv:1406.1078 (2014). [pdf] (First Seq-to-Seq Paper) :star::star::star::star:

   **[2]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014. [pdf] (Outstanding Work) :star::star::star::star::star:

   **[3]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv preprint arXiv:1409.0473 (2014). [pdf] :star::star::star::star:

   **[4]** Vinyals, Oriol, and Quoc Le. "A neural conversational model." arXiv preprint arXiv:1506.05869 (2015). [pdf] (Seq-to-Seq on Chatbot) :star::star::star:
   
   **[5]** Understanding LSTM Networks(http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
   
### Types of CNN
   **[0]** Dilated Convolutional Kernel
     - Fisher Yu, Vladlen Koltun:Multi-Scale Context Aggregation by Dilated Convolutions. ICLR(2016)
     
   **[1]** Deformable Convolutional Kernel
     - Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei:Deformable Convolutional Networks. CoRR abs/1703.06211 (2017)
     
   **[2]** Conventional Convolutional Kernel
    
### Types of Activation
   - sigmoid   
   - hard sigmoid
   - tanh
   - relu
   - lerelu
   - maxout
   - swish
   - softplus

   relu, lerelu, tanh, sigmoid is recommanded strongly!!!(https://medium.com/towards-data-science/activation-functions-neural-networks-1cbd9f8d91d6)
    
### Model Constraints
  **[0]** Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580 (2012). [pdf] (Dropout) :star::star::star:

  **[1]** Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from overfitting." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [pdf] :star::star::star:

  **[2]** Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015). [pdf] (An outstanding Work in 2015) :star::star::star::star:

  **[3]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016). [pdf] (Update of Batch Normalization) :star::star::star::star:

  **[4]** Courbariaux, Matthieu, et al. "Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1." [pdf] (New Model,Fast)  :star::star::star:

  **[5]** Jaderberg, Max, et al. "Decoupled neural interfaces using synthetic gradients." arXiv preprint arXiv:1608.05343 (2016). [pdf] (Innovation of Training Method,Amazing Work) :star::star::star::star::star:

  **[6]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [pdf] (Modify previously trained network to reduce training epochs) :star::star::star:

  **[7]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [pdf] (Modify previously trained network to reduce training epochs) :star::star::star:

  **[8]** Han, Song, Huizi Mao, and William J. Dally. "Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding." CoRR, abs/1510.00149 2 (2015). [pdf] (ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup) :star::star::star::star::star:

  **[9]** Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size." arXiv preprint arXiv:1602.07360 (2016). [pdf] (Also a new direction to optimize NN,DeePhi Tech Startup) :star::star::star::star:

### Optimization
#### Optimization Methods
  **[0]** 	Sebastian Ruder:An overview of gradient descent optimization algorithms. CoRR abs/1609.04747 (2016):star::star::star::star::star:(http://ruder.io/optimizing-gradient-descent/)
  
  **[1]** Back Propagation Algorithm(https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf)
 
  **[2]** Andrychowicz, Marcin, et al. "Learning to learn by gradient descent by gradient descent." arXiv preprint arXiv:1606.04474 (2016). [pdf] (Neural Optimizer,Amazing Work) :star::star::star::star::star:

#### Optimization Functions
   - Momentum
   - Nesterov accelerated gradient
   - Adagrad
   - Adadelta
   - RMSprop
   - Adam
   - AdaMax
   - Nadam

   :star::star::star::star::star:**Adam** is a better choice

# Journals and Periardical
  **Machine Learning and Theories** 
  - NIPS
  - ICML
  - ICLR
  
  **Computer Vision**
  
  - CVPR
  - ICCV
  - ECCV
  
  **Neural Language Processing**
  - EMNLP
  
  **Artifical Intelligence**
  - AAAI
  - IJCAI

# Public Accounts
  - 机器之心
  - 新智元
  
# Deep Learning Framework(open source framework)
  - Tensorflow(https://www.tensorflow.org/api_docs/)
  - Caffe(http://caffe.berkeleyvision.org/)
  - Pytorch(http://pytorch.org/docs/master/)
  - Keras(https://keras.io/)
  - Mxnet(https://mxnet.incubator.apache.org/get_started/)
  - etc.
 
# Other Sources
## Generative Adversarial Networks:(GAN):

   - GAN Paper
         https://github.com/hindupuravinash/the-gan-zoo
          
   - GAN Tricks
         https://github.com/soumith/ganhacks
          
   - GAN Codes
         Tensorflow:https://github.com/hwalsuklee/tensorflow-generative-model-collections
         Pytorch:https://github.com/znxlwm/pytorch-generative-model-collections

# New Architecture
  - Generative Adversarial Networks
  - Capsules(Dynamic Routing Between Capsules--by Hinton)
  
# References
 - https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap/blob/master/README.md
 - https://github.com/terryum/awesome-deep-learning-papers
 
# Contacts
 - Email:computerscienceyyz@163.com
