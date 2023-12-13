clear;clc;close all;

addpath(genpath('./'))
str = {'sample'};
for ii = 1:length(str)
    load(str{ii});
    Y = train_target;
    Y(train_target<0) = 0;
    %Normalization is recommended for datasets with very small values of features, otherwise, label enhancement will have little effect.
    train_data = normalize(train_data,'range');
    test_data = normalize(test_data,'range');
    i = 1;
    
    %%%%%%%%%%%%%%%Shared data
    [n,d] = size(train_data);
    
    W_s   = ones(n, n)*.5;
    W_s_1 = W_s;

    iter  = 1;
    oldloss = 0;
    bk = 1; bk_1 = 1; 
    alpha = 1;

    XXT = train_data*train_data';
    Lip = norm(XXT,'fro');
    t = alpha/Lip;
    maxIter = 1000;
    minimumLossMargin = 1e-2;

    while iter <= maxIter
        W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
        Gw_s_k = W_s_k - t * gradient(XXT,W_s_k);
        bk_1   = bk;
        bk     = (1 + sqrt(4*bk^2 + 1))/2;

        W_s_1  = W_s;   

        W_s    = softthres(Gw_s_k,t);
        W_s(logical(eye(size(W_s)))) = 0;

        totalloss = .5*(norm((W_s*train_data - train_data), 'fro'))^2+alpha*norm(W_s,1);

        if abs(oldloss - totalloss) <= minimumLossMargin
            break;
        else
            oldloss = totalloss;
        end
        iter=iter+1;
    end

    S = W_s;
    %%%%%%%%%%%%%%%

    for lambda1 = [0,0.01,0.1,1,10,100]
        for lambda2 = [0.01,0.1,1,10,100]
%             lambda1=0.01;
%             lambda2=1;
            lambda3 = 1;
            HammingLoss(i,1) = lambda1;
            HammingLoss(i,2) = lambda2;
            RankingLoss(i,1) = lambda1;
            RankingLoss(i,2) = lambda2;
            OneError(i,1) = lambda1;
            OneError(i,2) = lambda2;
            Coverage(i,1) = lambda1;
            Coverage(i,2) = lambda2;
            Average_Precision(i,1) = lambda1;
            Average_Precision(i,2) = lambda2;
            
           
            [feature_slct, feature_weight_sorted] = CLEMLFS(train_data,Y',lambda1,lambda2, S);

            numFeature = size(train_data,2);
            if numFeature>1000
                numSeleted = round(numFeature * 0.1);
            elseif numFeature<=1000 && numFeature>500
                numSeleted = round(numFeature * 0.2);
            elseif numFeature<=500 && numFeature>100
                numSeleted = round(numFeature * 0.3);
            else
                numSeleted = round(numFeature * 0.4);
            end
            
            selFeature = feature_slct(1:numSeleted);
            Num=10;Smooth=1;
            [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,selFeature),train_target,Num,Smooth);
            [HammingLoss(i,3),RankingLoss(i,3),OneError(i,3),Coverage(i,3),Average_Precision(i,3),Outputs,Pre_Labels]=MLKNN_test(train_data(:,selFeature),train_target,test_data(:,selFeature),test_target,Num,Prior,PriorN,Cond,CondN);
            i=i+1;
        end
    end
    
    filename = ['CLEMLFSAllParameter ' str{ii}];
    save(filename, 'HammingLoss', 'RankingLoss','OneError','Coverage','Average_Precision');
    clear HammingLoss;
    clear RankingLoss;
    clear OneError;
    clear Coverage;
    clear Average_Precision;
end

function W = softthres(W_t,thres)
    W = max(W_t-thres,0) - max(-W_t-thres,0);
end

function gradientvalue = gradient(XXT,W)
    gradientvalue = W*XXT-XXT;
end