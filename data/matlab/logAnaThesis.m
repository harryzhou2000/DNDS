
prefix = "../recv/out/";
logNames = {
    prefix + "test_0.log";
    prefix + "test_1.log";
    prefix + "test_2.log";
    prefix + "test_3.log";
    prefix + "test_4.log";
    prefix + "test_5.log";
    prefix + "test_6.log";
    prefix + "test_7.log";
    prefix + "test_8.log";
    prefix + "test_9.log";
    prefix + "test_6_45.log";
    prefix + "test_7_45.log";
    prefix + "test_8_45.log";
    };

results = cell(numel(logNames), 1);


for i = 1:numel(logNames)
    [results{i}.s,results{i}.r,results{i}.e,results{i}.em,results{i}.ConvStep,results{i}.E,results{i}.EM]=...
        FlogAna(logNames{i}, 1.1);
end

for i = 1:numel(logNames)
    fprintf("%s, %.2E & %.2E & %6d\\\\\n",...
        logNames{i},...
        results{i}.E,...
        results{i}.EM,...
        results{i}.ConvStep);
    
end

figure(112);
MD = 10;
LW = .5;
clf; hold on;
isee = 7;
plot(results{isee}.s,results{isee}.r,'-x','DisplayName','Scheme 6','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
isee = 9;
plot(results{isee}.s,results{isee}.r,'-^','DisplayName','Scheme 8','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
isee = 11;
plot(results{isee}.s,results{isee}.r,'-o','DisplayName','Scheme 6 $(45^\circ)$','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
isee = 13;
plot(results{isee}.s,results{isee}.r,'-d','DisplayName','Scheme 8 $(45^\circ)$','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
L = legend;
L.Interpreter = 'latex';
set(gca,'FontName','Times New Roman');
set(gca,'YScale','log');
xlabel('iteration');
ylabel('$\left\|RHS\right\|_{1}$', 'Interpreter' , 'latex');
grid on; grid minor;
xlim([0,500]);
set(gcf,'Position',[100,100,400,380]);
saveas(gcf, '68_Res','epsc');

figure(113);
MD = 10;
LW = .5;
clf; hold on;
isee = 7;
plot(results{isee}.s,results{isee}.e,'-x','DisplayName','Scheme 6','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
isee = 9;
plot(results{isee}.s,results{isee}.e,'-^','DisplayName','Scheme 8','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
isee = 11;
plot(results{isee}.s,results{isee}.e,'-o','DisplayName','Scheme 6 $(45^\circ)$','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
isee = 13;
plot(results{isee}.s,results{isee}.e,'-d','DisplayName','Scheme 8 $(45^\circ)$','MarkerIndices',1:MD:numel(results{isee}.r),'LineWidth',LW);
L = legend;
L.Interpreter = 'latex';
set(gca,'FontName','Times New Roman');
set(gca,'YScale','log');
xlabel('iteration');
ylabel('$E_1$ Error', 'Interpreter' , 'latex');
grid on; grid minor;
xlim([0,500]);
set(gcf,'Position',[100,100,400,380]);
saveas(gcf, '68_Err','epsc');
