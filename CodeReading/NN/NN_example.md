
# 关于Example: Neural Network的解读

参考matlab的[DeepLearningToolBox](https://github.com/rasmusbergpalm/DeepLearnToolbox)

## 训练数据

训练数据为mnist_data.mat，加载`load mnist_unit8.mat`。mnist_data.mat中，有四个数据：

- train_x (60000x784 unit8)
- train_y (60000x10 unit8)
- test_x (10000x784 unit8)
- test_y (10000x784 unit8)

由于在MNIST数据集中，训练集有60000个examples，测试集有10000个examples，图像大小为28x28（其中28x28=784）。因此可以得到，train_x中每一行数据代表一个example，而train_y代表train_x中每个example对应的类别（0-9总共10个类别）

`train_x = double(train_x) / 255`将train_x归一化到[0,1]

`[train_x, mu, sigma] = zscore(train_x);`使用公式 Y=(X-mean(X))./std(X)处理train_x，其中mu为mean，sigma为std。

- mu (1x784 double)
- sigma (1x784 double)

mu和sigma是在每个像素点的位置，通过对所有样本的计算，得到的数值，所以其有784个值

`test_x = normalize(test_x, mu, sigma)`用mu和sigma处理规范化处理test_x

### normalize函数

    function x = normalize(x, mu, sigma)
        x=bsxfun(@minus,x,mu);
            x=bsxfun(@rdivide,x,sigma);
    end

其中normalize函数位于util目录下的normalize.m中，其中bsxfun是对矩阵点进行位处理的，bsxfun(@minus,x,mu)是将mu(1x784)复制为与x相同维数(60000x784)，然后将x减去mu；同理，x=bsxfun(@rdivide,x,sigma)是将sigma(1x784)复制为与x相同维数(60000x784)，然后将x除以sigma

`rand('state',0)`表示随机产生数的状态state，一般情况下不用指定状态。但是有时候为了显示与例子相同的结果，采用了设置state，rand('state',0)作用在于如果指定状态，产生随机结果就相同了

`nn=nnsetup([784 100 10])` NNSETUP创建了前向和后向的神经网络，返回nn=numel(architecture)，即返回数组architecture中元素的个数

### nnsetup函数

    function nn=nnsetup(architecture)
    
        nn.size = architecture      // 即nn.size为[784 100 10]
        
        nn.n = numel(nn.size)       //即nn.n=nemel([784 100 10])，返回数组中元素个数，nn.n为3
        
        nn.activation_function = ‘tan_opt’ //activation函数使用tanh函数，也可使用'sigm'即sigmoid
        
        nn.learningRate = 2     // 学习率为2，使用sigmoid函数和非归一化输入时，值需要小些
        
        nn.momentum = 0.5
        
        nn.scaling_learningRate             //Scaling factor for the learning rate (each epoch)
        nn.weightPenaltyL2 = 0            //L2 regularization
        nn.nonSparsityPenalty = 0       //Non sparsity penalty
        nn.sparsityTarget = 0.05          //Sparsity target
        nn.inputZeroMaskedFraction = 0      //Used for Denoising AutoEncoders
        nn.dropoutFraction = 0            //Dropout level
        nn.testing = 0                        //Internal variable. nntest sets this to one.
        nn.output = 'sigm'                // output unit 'sigm' (=logistic), 'softmax' and 'linear'

        for i=2: nn.n
        
            nn.W{i-1}=(rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));     //{}用于cell型数组的定义和引用，可以用nn.W{1}和nn.W{1}表示第一层连接参数和第二层连接参数；
            //在此W<1x2 cell>为cell型数组，其中只存储<100x785>与<10x101 double>这两个信息，即W{1}为<100x785 double>，即<100x(784+1)>，其中存储了参数具体值；W{2}为<10x101 double>，即<10x(101+1)>，其中存储了参数具体值。

            nn.vW{i - 1} = zeros(size(nn.W{i - 1}));        //vW的大小与W相同，只不过其值全为0

            nn.p{i} = zeros(1, nn.size(i));                 //average activations，用于sparsity

        end

`opts.numepochs =  1`定义了循环次数；而`opts.batchsize = 100`定义计算每次mean gradient的样本个数

`[nn, L] = nntrain(nn, train_x, train_y, opts)`

### nntrain函数

    function [nn, L] = nntrain(nn, train_x, train_y, opts, val_x, val_y)

    // NNTRAIN训练神经网络，输入x，输出y，设置opts.numepochs epochs与minibatches of size opts.batchsize。返回updated activations, error, weights与biases的nn网络（结构体），以及各个traininig nimibatch的均方差之和，L（结构体）

        assert(isfloat(train_x), 'train_x must be a float');    //判断train_x是否为float型，如果不是，返回error 'train_x must be a float'     

        assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')    //判断nntrain函数输入的变量，nargin为number of input arguments

        loss.train.e  = [];          
        loss.train.e_frac = [];
        loss.val.e = [];
        loss.val.e_frac = []; 

        opts.validation = 0;        //对验证集的判断
        
        if nargin == 6                  //验证集的使用，在有验证集的情况下，设置validation为1
           opts.validation = 1;  
        end

        fhandle = [];
        if isfield(opts,'plot') && opts.plot == 1 //是否需要画图表示
            fhandle = figure();    
        end

        m = size(train_x, 1);                       //m为train_x的行数，即为样本个数

        batchsize = opts.batchsize;             //每次计算的batchsize为100

        numepochs = opts.numepochs;         //epoch次数为1

        numbatches = m / batchsize;             //样本中batch的个数，为600个batch

        assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');         //rem函数的求整除的函数，判断batch个数必须为整数

        L = zeros(numepochs*numbatches,1);           //每个batch计算存储一次平均loss值，即计算存储了600个batch的loss值，共计算numepoch次，即1次。L为<600x1 double>
        
        n=1;
        for i = 1 : numepochs           //计算迭代次数
            tic                             //用来保存当前时间
            
            kk = randperm(m);       //随机打乱一个数字序列，即kk为<1x60000 double>，其中顺序随机
            for l = 1 : numbatches          //计算次数为batch的个数
                batch_x = train_x(kk((l-1)*batchsize + 1: l*batchsize), :);