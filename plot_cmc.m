function cmc = plot_cmc(score_mat,test_labels)
% calculates the rank-k recognition rate 
% for for k = 1:n_models, using score matrix. 
% Inputs: 
%       score_mat: n_test x n_models matrix of scores
%                  n_test is the number of queries
%                  n_models is the number of models (individual classes)
%                  higher scores means better match. 
%       test_labels: n_test x 1 vector containing true query labels
% 
% Output
%       cmc: Cumulative Match Curve
%
% Navid Shokouhi, 2018

[n_test,n_models] = size(score_mat);
true_mat = zeros(n_test,n_models);
for i = 1:n_test
    true_mat(i,test_labels(i)) = 1;
end

cmc = zeros(1,n_models);
for k = 1:n_models
    mx = max(score_mat,[],2); 
    true_mat_est = bsxfun(@eq,score_mat,mx);
    if k == 1 
        cmc(k) = sum(sum(true_mat.*true_mat_est))/n_test; 
    else
        cmc(k) = min(100,sum(sum(true_mat.*true_mat_est))/n_test + cmc(k-1)); 
    end
    score_mat(true_mat_est) = -1;
end
end
