close all
clear

Array=csvread('samples_cufft.csv',1,0);

x = Array(:, 1);
y=Array(:, 2);
z= Array(:, 3);
real_val = Array(:, 4);
imag_val = Array(:, 5);

figure
scatter3(x,y,z,5,real_val,'filled')
colorbar
saveas(gcf,'cufft_plot.png')


close all
clear

Array=csvread('samples_fftw.csv',1,0);

x = Array(:, 1);
y=Array(:, 2);
z= Array(:, 3);
real_val = Array(:, 4);
imag_val = Array(:, 5);

figure
scatter3(x,y,z,5,real_val,'filled')
colorbar
saveas(gcf,'fftw_plot.png')
