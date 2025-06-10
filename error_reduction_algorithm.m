clearvars
close all
clc

max_iter = 1000;
load("function3.mat"); f = image;
N = size(f,1); M = 2 * N;
x = linspace(-1,1,N); y = x;

figure()
subplot(1,2,1)
imagesc(x,y,real(f))
title("\textbf{Real part}","Interpreter","latex")
set(gca,'dataAspectRatio',[1 1 1])
axis off

subplot(1,2,2)
imagesc(x,y,imag(f))
title("\textbf{Imaginary part}","Interpreter","latex")
set(gca,'dataAspectRatio',[1 1 1])
axis off

sgtitle("\textbf{Original image}","Interpreter","latex")

F = fft2(f,M,M);
z = rand(N) .* exp(1i * 2*pi * rand(N));
counter = 0;
error = zeros(1,max_iter);

while counter < max_iter

    Z = fft2(z,M,M);
    Z = abs(F) .* exp(1i * angle(Z));
    z = ifft2(Z,"symmetric"); z = z(1:N,1:N);      % add "symmetric" as an input for ifft2 when working with real signals
    z = prior(z);

    counter = counter + 1;
    error(counter) = norm(abs(F) - abs(fft2(z,M,M)),'fro') / norm(F,'fro');

end

z = z * conj(trace(f'*z)) / norm(z,"fro")^2;  % Align with the global phase of f

figure()
loglog(error)
title("\textbf{Relative global error in the Fourier space}","Interpreter","latex")
xlabel("Iteration","Interpreter","latex")

figure()
subplot(1,2,1)
imagesc(x,y,real(z))
title("\textbf{Real part}","Interpreter","latex")
set(gca,'dataAspectRatio',[1 1 1])
axis off

subplot(1,2,2)
imagesc(x,y,imag(z))
title("\textbf{Imaginary part}","Interpreter","latex")
set(gca,'dataAspectRatio',[1 1 1])
axis off

sgtitle("\textbf{Reconstructed image}","Interpreter","latex")
