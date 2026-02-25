function [Pxx,f] = psd_c(x, Fs, N)

w = hann(N);
Pxx = pwelch (x, w, []); 
% Pxx=fftshift(Pxx);
Pxx = 10*log10(fftshift(Pxx));
Pshift=0-max(Pxx);
Pxx=Pxx+Pshift;
f = Fs*((0:length(Pxx)-1)/length(Pxx) - 0.5)/1e6;

% f=1:N;
% plot(f, Pxx,type,'linewidth',2);
plot(f, Pxx,'linewidth',2);

% set(gca,'Ytick',-60:10:0,'FontSize',22,'FontWeight','b','LineWidth',2,'FontName','Calibri');
% axis([-100 100 -60 0])
% xlabel('Frequency Offset(MHz)');
% ylabel('Power Spectral Density(dB/Hz)');
% set(Axes,);

% xlabel('Frequency Offset(MHz)','FontSize',27,'FontWeight','b','FontName','Calibri');
% ylabel('Normalized PSD(dBm/Hz)','FontSize',27,'FontWeight','b','FontName','Calibri');

% set(gca,'position',[0.158935185185185,0.167977963073258,0.809293981481481,0.799999999999999]);
% set(gcf,'unit','normalized','position',[0.46,0.098148148148148,0.45,0.621851851851852]);
%  set(gcf,'InnerPosition',[0.158458499570021,0.172168527178692,0.778827278480089,0.795502515930059]);

return;
