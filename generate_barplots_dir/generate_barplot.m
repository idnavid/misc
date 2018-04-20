close all
data    = [0.59	0.65	0.67	0.70	0.73
           0.17	0.14	0.10	0.09	0.08
           0.19	0.18	0.14	0.12	0.10
           0.12	0.14	0.13	0.13	0.11];
data = data';
methods = {'method1','method2','method3','method4'};

x_values=4:2:12;
B = bar(x_values, data,'grouped');
set(gca,'FontSize',12)
set(B,'LineWidth',1.5)
legend(methods,'Location','northeast')
ylabel('Y label')
xlabel('x label')
applyhatch_pluscolor(gcf, '+/cwk');

