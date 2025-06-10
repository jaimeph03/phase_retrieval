clearvars
close all
clc

image = load("face9.mat","image").image;

figure()
subplot(1,2,1)
imagesc(real(image))
title("\textbf{Real part}","Interpreter","latex")
set(gca,'dataAspectRatio',[1 1 1])
axis off

subplot(1,2,2)
imagesc(imag(image))
title("\textbf{Imaginary part}","Interpreter","latex")
set(gca,'dataAspectRatio',[1 1 1])
axis off

sgtitle("\textbf{Original image}","Interpreter","latex")

[mcoh,time,error,recovered_x] = nesterov_agdm(image);

figure()
subplot(1,2,1);
imagesc(real(recovered_x))
set(gca,'dataAspectRatio',[1 1 1])
axis off
title('\textbf{Real part}',"Interpreter","latex")

subplot(1,2,2);
imagesc(imag(recovered_x))
set(gca,'dataAspectRatio',[1 1 1])
axis off
title('\textbf{Imaginary part}',"Interpreter","latex")

sgtitle("\textbf{Reconstructed image}","Interpreter","latex")

fprintf('Execution time: %.3f\n', time)
fprintf('Relative error: %.3e\n\n', error)
