close all; clearvars

% Support vector machine
load('rbf_svm_cv_mean_acc.mat')
figure(1)
gamma = linspace(-1,-3,3);
C = linspace(3.5,5,10);
imagesc(C,gamma,cv_mean_acc)
c = colorbar;
c.Label.String = 'Mean accuracy';
title('Cross Validation: Support Vector Machine')
xlabel('C')
ylabel('log_{10}\gamma')

figure(2)
plot(C,cv_mean_acc(2,:))
title('Cross Validation: Support Vector Machine')
xlabel('C')
ylabel('Mean accuracy')

% k-nearest neighbors
load('knn_cv_mean_acc.mat')
k = 1:10;
figure(3)
hold on
plot(k,cv_mean_acc(1,:))
plot(k,cv_mean_acc(2,:))
hold off
title('Cross Validation: k-Nearest Neighbors')
xlabel('k')
ylabel('Mean accuracy')
legend('weights = ''uniform''','weights = ''distance''')

% Random forest
load('rnd_cv_mean_acc.mat')
n_trees = 5*logspace(2,3,10);
figure(4)
semilogx(n_trees,cv_mean_acc)
title('Cross Validation: Random Forest')
xlabel('n_{trees}')
xlim([5e2 5e3])
ylabel('Mean accuracy')
