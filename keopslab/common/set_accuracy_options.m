function options = set_accuracy_options(options,formula)

% Set accuracy options for summations

% use_double_acc: accumulate results of reduction in double instead of float ?
options = set_default_option(options,'use_double_acc',0);

% sum_scheme: method used for summation: This can only be set for
% summation type reductions. May be either:
% - 'direct_sum': direct summation
% - 'block_sum': accumulate results in each block before summing to output.
% This option is enabled by default because
% it improves accuracy with no overhead.
% - 'kahan_scheme': use Kahan scheme for compensating round-off errors.

% We check that reduction is of summation type
if strcmp(formula(1:12),'GradFromPos(')
    ind_start = 13;
else
    ind_start = 1;
end
ind_end = strfind(formula,'_Reduction(') - 1;
is_sum_type_red = any(strcmp(formula(ind_start:ind_end),...
    {'Sum','MaxSumShiftExp','MaxSumShiftExpWeight'}));
if is_sum_type_red
        default_sum_scheme = 1; % internally 1 means block_sum 
else
        default_sum_scheme = 0;
end
options = set_default_option(options,'sum_scheme',default_sum_scheme);
% another test to check that user has not set this option wrongly for
% non-summation reductions
if ~is_sum_type_red && options.sum_scheme~=0
    error('Block sum or Kahan summation can only be used for summation type reductions')
end
