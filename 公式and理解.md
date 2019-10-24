rnn&lstm推导的理解

1. 时序性的理解
   ht的计算通过ht-1得到，而ht则参与得到了ht+1，所以在求ht导数的时候，需要知道ht+1导数部分，才能够得到结果。
2. 公式部分要清晰，可以从实现的角度来推导，时序涉及前一时刻的项目是ht-1，那么需要计算偏loss/ht，就要考虑两个部分，第一部分，是ht参与了后面层的运算，第二部分是ht参与了ht+1的运算，所以导数由两部分组成
3. 一个推导的很好的链接，虽然有些错误，但是总体是正确的
4. https://manutdzou.github.io/2016/07/11/RNN-backpropagation.html

![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/F5D0B7CF-AA87-43F3-B3BE-345720A5B0EC.png)

2. 交叉熵、focal-loss、softmax、hinge loss

   ![img](file:///C:/Users/x00508557/AppData/Roaming/eSpace_Desktop/UserData/x00508557/imagefiles/3048ABA1-4726-45B1-B889-34EC718FEF6F.png)

   

   softmax，softmax是将logits变成prob的转换，求解loss还是使用交叉熵
   $$
   yj = exp(zj)/sum(exp(zi))
   $$

   $$
   loss = sum(log(yj)*labelj)
   $$

   注意，计算softmax 交叉熵loss的时候，只有一个类别是1.0，其他都是0

   