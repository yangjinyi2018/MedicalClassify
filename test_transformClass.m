addpath(genpath('E:\study\MachineLearning\up\scatnet'))
addpath(genpath('E:\study\MachineLearning\up\libsvm-3.17'))

%加载经mixup后的图像数据
load('generateDone.mat');
load('label_train');

%图像特征变换的参数
filt_opt.L1 = 2;
filt_opt.L2 = 2;
filt_opt.J = 3;
scat_opt.M = 2;

%提取训练集特征
for i=1:465
    data=reshape(double(train(i,:,:,:))/255,[32,32,32]);
    img=data;
    %设置散射变换的参数   
    Wop = my_wavelet_factory_3d(size(img),filt_opt, scat_opt);
    Sx = scat(img, Wop);
    %进行变换，得到子带
    S = format_scat(Sx);
    for j=1:37
        SS=S(j,:,:,:);
        EE=SS(:).^2;
        %feature1 均值
        F1(j)=mean(SS(:));
        %feature2 能量
        F2(j)=norm(SS(:),2)^2;
        %feature3 标准差
        F3(j)=std(SS(:));
        %feature4 熵
        p=EE./sum(EE);
        F4(j)=-sum(p.*log2(p));
    end
     X_train(i,:)=[F1 F2 F3 F4];
end

%提取验证集特征
for i=1:117
    data=reshape(double(test(i,:,:,:))/255,[32,32,32]);
    img=data;    
    Wop = my_wavelet_factory_3d(size(img),filt_opt, scat_opt);
    Sx = scat(img, Wop);
    S = format_scat(Sx);
    for j=1:37
        SS=S(j,:,:,:);
        EE=SS(:).^2;
        F1(j)=mean(SS(:));
        F2(j)=norm(SS(:),2)^2;
        F3(j)=std(SS(:));
        p=EE./sum(EE);
        F4(j)=-sum(p.*log2(p));
    end
        X_test(i,:)=[F1 F2 F3 F4];
end
  
%训练集标签
Y_train=label;
X_train(253,:)=X_train(252,:);
Y_train(253)=Y_train(252);

%归一化特征
for i=1:148
    X_train_norm(:,i)=(X_train(:,i)-mean(X_train(:,i)'))/std(X_train(:,i)');

    X_test_norm(:,i)=(X_test(:,i)-mean(X_train(:,i)'))/std(X_train(:,i)');
end

m=1

len = 465;

load('M1.mat');

%验证集
X_test2=double(X_test_norm);

disp('验证集')
[predict_label, accuracy, prob2] = svmpredict(zeros(117,1),X_test2, model_SVM,'-b 1');
 
probb(:,m)=prob2(:,1);

m=2
%训练集标签
Y_train=label;
Y_train(253)=Y_train(252);

load('M2.mat')

X_test2=double(X_test_norm);

disp('验证集')
[predict_label, accuracy, prob2] = svmpredict(zeros(117,1) ,X_test2, model_SVM2,'-b 1');
 
%集成模型的分类结果
probb(:,m)=prob2(:,1);
prob=mean(probb')';
 
%生成csv表格

name=["candidate11"
"candidate13"
"candidate15"
"candidate17"
"candidate22"
"candidate26"
"candidate33"
"candidate40"
"candidate42"
"candidate49"
"candidate56"
"candidate59"
"candidate60"
"candidate68"
"candidate75"
"candidate76"
"candidate77"
"candidate79"
"candidate85"
"candidate99"
"candidate100"
"candidate103"
"candidate107"
"candidate108"
"candidate112"
"candidate116"
"candidate117"
"candidate118"
"candidate128"
"candidate129"
"candidate132"
"candidate137"
"candidate144"
"candidate146"
"candidate149"
"candidate165"
"candidate167"
"candidate170"
"candidate174"
"candidate178"
"candidate189"
"candidate191"
"candidate194"
"candidate195"
"candidate197"
"candidate201"
"candidate204"
"candidate223"
"candidate239"
"candidate251"
"candidate252"
"candidate259"
"candidate264"
"candidate268"
"candidate274"
"candidate275"
"candidate278"
"candidate284"
"candidate286"
"candidate290"
"candidate300"
"candidate301"
"candidate302"
"candidate309"
"candidate312"
"candidate329"
"candidate333"
"candidate344"
"candidate346"
"candidate348"
"candidate354"
"candidate357"
"candidate360"
"candidate363"
"candidate368"
"candidate373"
"candidate384"
"candidate390"
"candidate394"
"candidate396"
"candidate397"
"candidate398"
"candidate402"
"candidate403"
"candidate412"
"candidate424"
"candidate427"
"candidate431"
"candidate439"
"candidate440"
"candidate450"
"candidate451"
"candidate455"
"candidate456"
"candidate458"
"candidate459"
"candidate463"
"candidate466"
"candidate483"
"candidate486"
"candidate489"
"candidate491"
"candidate493"
"candidate498"
"candidate499"
"candidate502"
"candidate509"
"candidate513"
"candidate524"
"candidate530"
"candidate544"
"candidate556"
"candidate563"
"candidate564"
"candidate565"
"candidate580"
"candidate582"
];
columns = {'name','predicted'};
data = table(name,prob, 'VariableNames', columns);
writetable(data, 'Submission.csv')
