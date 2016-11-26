function plot_errorbar_param_conv(input,param,conditions, ylim)
hold on
for i = 1:length(input)
    res = input{i};
    errorbar(mean(res'),std(res'))
end
set(gca,'XTick',1:length(param))
set(gca,'XTickLabel',cellfun(@num2str, num2cell(param), 'un',0))
xlabel('beta')
ylabel('number of steps to convergence')
axis([0,length(param)+1,ylim(1),ylim(2)])
legend(conditions)
end