clearvars
close all
clc

rng(1)
max_iter = 10000;
f = load("complex1.mat","image").image;
N = size(f,1);
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

F = fft2(f);
z = abs(f) .* exp(1i * 2*pi * rand(N));
counter = 0;
error_real = zeros(1,max_iter);
error_imag = zeros(1,max_iter);
error = zeros(1,max_iter);

while counter < max_iter

    Z = fft2(z);
    Z = abs(F) .* exp(1i * angle(Z));
    z = ifft2(Z);
    z = abs(f) .* exp(1i * angle(z));
    z = z * exp(1j * (angle(f(10,10)) - angle(z(10,10))));
    
    counter = counter + 1;
    error_real(counter) = norm(real(F) - real(fft2(z)),'fro') / norm(real(F),'fro');
    error_imag(counter) = norm(imag(F) - imag(fft2(z)),'fro') / norm(imag(F),'fro');
    error(counter) = norm(abs(F) - abs(fft2(z)),'fro') / norm(F,'fro');

end

figure()
subplot(2,1,1); hold on; box on
plot(error_real); plot(error_imag)
set(gca,'XScale','log'); set(gca,'YScale','log')
title("\textbf{Relative error in the real and imaginary parts of the signal}","Interpreter","latex")
xlabel("Iteration","Interpreter","latex")
legend("Real part","Imaginary part")

subplot(2,1,2)
loglog(error)
title("\textbf{Relative global error in the Fourier space}","Interpreter","latex")
xlabel("Iteration","Interpreter","latex")

