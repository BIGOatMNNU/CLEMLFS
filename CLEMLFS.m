%Normalization is recommended for datasets with very small values of features, otherwise, label enhancement will have little effect.
function [feature_slct, feature_weight_sorted, time] = CLEMLFS(data, target, lambda1, lambda2, S)
start=tic;

%data=normalize(data,'range');
[n,d] = size(data);
[~,c] = size(target);
%%label enhancement
%Gaussian kernel matrix
ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
par  = 1*mean(pdist(data)); %parameter of kernel function
H = kernelmatrix(ker, par, data, data);% build the kernel matrix on the labeled samples (N x N)
UnitMatrix = ones(size(data,1),1);
trainFeature = [H,UnitMatrix];
clear par;
clear H;
% 

%Obtaining relationships between samples (proximal gradient descent)
if nargin < 5

    W_s   = ones(n, n)*.5;
    W_s_1 = W_s;

    iter  = 1;
    oldloss = 0;
    bk = 1; bk_1 = 1; 
    alpha = 1;

    XXT = data*data';
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

        totalloss = .5*(norm((W_s*data - data), 'fro'))^2+alpha*norm(W_s,1);

        if abs(oldloss - totalloss) <= minimumLossMargin
            break;
        else
            oldloss = totalloss;
        end
        iter=iter+1;
    end

    S = W_s;
end
%Label enhancement optimization process
temp1 =  trainFeature'*trainFeature;
temp2 = lambda1*(S*trainFeature-trainFeature)'*(S*trainFeature-trainFeature)+temp1;
clear temp1;


W = pinv(temp2)*(trainFeature'*target);

distribution = trainFeature*W;
% distribution = softmax(distribution')';
for i = 1:n
    distribution(i,:) = distribution(i,:) - min(distribution(i,:));
    distribution(i,:) = distribution(i,:)/sum(distribution(i,:));
end

theta = ones(d, c)*.5;


%Feature Selection Section
iter=1;
eps = 1e-10;
while(1)
    D = diag(1./max(sqrt(sum((theta).*(theta),2)),eps));
    theta = (data'*data+lambda2*D)\(data'*distribution);
    
    obj(iter) = .5*(norm((data*theta - distribution), 'fro'))^2+lambda2*sum(sqrt(sum((theta).*(theta),2)));
    disp(iter+":"+obj(iter));
    if iter>=2 && abs(obj(iter)-obj(iter-1))<=1e-3
        break;
    end
    iter=1+iter;
end
% theta = theta - min(theta,[],[1,2]);

%%Feature ordering section
feature_weight = mean(abs(theta), 2);
[feature_weight_sorted, value] = sort(feature_weight, 'descend');
feature_slct = value(1:end);

time = toc(start);
end

function W = softthres(W_t,thres)
    W = max(W_t-thres,0) - max(-W_t-thres,0);
end

function gradientvalue = gradient(XXT,W)
    gradientvalue = W*XXT-XXT;
end
