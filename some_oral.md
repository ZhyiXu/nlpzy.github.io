ner task using bert language model with crf help the intelligent meeting system in hw to realize the function of ner and intent recognition

and i also in charge of the communication jobs with other department, because our lab has to output some achivement to other 业务方，these called接口工作



bert is a language model that consists of many transformer units and transformer is consist of self-attention unit and feed forward neural network.

attention is the key of transformer, for a sentence it can focus on multi point no matter forward or backward.



the principle of line. line is short for large-scale information network embedding. it is an embedding algorithm base on graph data. computation of first-order similarity and second-order similarity.

first-order proximity is the similarity between two nodes, usually used weight to represent the proximity between two nodes. if there is no connection between two nodes the proximity is zero.

hower, this kind of representation is not enough sometime ,because maybe there is no connection between two nodes, the proximiyt is still very high, maybe because they share the same neighbor or they are belong to the same group. so there comes the second-order proximity. this proximity is the similarity between two groups connecting with each node.

first-order proximity: the joint prob is that first define the joint prob between to nodes. use sigmoid function the metric the similarity or proximity. the weight needs to be normalized. and finally the evaluate the distance between two prob distribution.

Kullback–Leibler divergence is always used two metric the difference between two prob distributions. so the target function to opitimize is below function.

![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/827F74E8-F6B8-4C63-A522-296858224673.png)  

And then we need to calculate the second-order proximity

it suits directed graph or undirected graph. the meaning of second-order proximity is that except for itself representation, it also include the context information. if the context information of two nodes are similar, these two nodes are similar.

![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/7BCC53BA-8B70-4EEE-B5C1-83F41D1B4C56.png)

the prob of the context node or neighbor of vj of node vi is p2

some opitimization skills is negative sampling and edge sampling

other embedding methods:node2vec, deepwalk

comparation: node2vec need the feature of node ,because the data we used were on the server of other group we only had the right to read the data, we did not have the right to copy the data to our server, the data transition was a big problem. as for deepwalk because the trival of line is not very effective so we didnot try deepwalk, the problem that the algorithm was not work, from my view and anylization is the data itself, we only used first-degree data , but in creidt risk scenario, this is not sufficient, and the graph we got is very sparse. the conection was not enough also. but if we use more, like second degree relation the data would explode to 20 million, that was to much for computation due the limit of compuatation resource.

------

graph convolutional network

as for the structual data like graph, relations traditional convolutional network or recurrent network is not suitable again.

many important data in real world is graph， like social network world wide web，knowledge graph。

first we need to get the adjacent matrix.

filter parameters are typically shared over all locations in graph. like in picture

the input includes an adjacency matrix and a feature matrix.

and also we need to build D diagonal node degree matrix.

multiplication with A means that, for every node, we sum up all the feature vectors of all neighboring nodes, with the existence of D feature of itself also included.

then :

![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/08F8979C-C696-4CEF-9A85-FFB62B919BD4.png)

also:

![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/147C5132-6EA8-440A-9496-D377693BF022.png)

![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/51AC7F98-A5C1-4C38-985F-D670395583A1.png)

about my task

actually, it was just an exploration, it was not a proper method for our task, the most problem was the we could not get the feature data of each node. what was the more important was that the overlap problem, the first degree relation of the train samples could not cover the test samples, yeah the user of our product only occupied a small propotion of the wechat users.

------

bert bidirectional encoder representations from transformers

nlp pretrained model. bert has outperformed several models in nlp

gpt -- generalized language model

two strategies:

mask language model , next sentence prediction

architecture； multi-layer transformers

------

about asr

automatic speech recognition

do some job to increased the robustness of ocr and know the whole process of asr

first acoustic model

the input is audio data  and then the audio should be cut into many pieces, each piece may be with 12 milie seconds  less or more, and then extract feature from the each piece, and then to do the **phoneme** recognition

CTC(Connectionist Temporal Classification)

ctc to solve the alignment between output and label in ocr and asr 

language model

language model to match phoneme to word and to sentence





