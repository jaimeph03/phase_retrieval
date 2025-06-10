function [mcoh,time,error,recovered_x] = proximal_gdm(image,delta)

tic
n = size(image,1);
x = image(:);

masks = 4;
d = randi([0 1],masks,n^2);
F = dftmtx(n);
A = kron(F,F);
for i = 1:masks-1
    W = diag(d(i,:));
    A = [A; kron(F,F) * W];
end
m = size(A,1);
y = abs(A * x).^2;

% Additive noise to intensity data
if nargin == 1; delta = 0; end
noise = delta * abs(y) .* randn(m,1);
y = y + noise;

% Mutual coherence of the measurement vectors
mcoh = abs(A' * A);
mcoh = mcoh - diag(diag(mcoh));

max_iter = 15000;
tau = 0.005;            % thresholding parameter
X = zeros(n^2);
res = zeros(max_iter,1);

Amap = @(X) diag(A * X * A');
ATmap = @(y) A' * diag(y) * A;
f = @(X) norm(Amap(X) - y)^2 / 2;
g = @(X) ATmap(Amap(X) - y);

L = real(eigs(@(y) Amap(ATmap(y)), m, 1));
beta = 1 / L;
    
for i = 1:max_iter

    fx = f(X); gx = g(X);
    res(i) = fx;
    X = X - beta * gx;
    [U,S,V] = svd(X); S = diag(max(diag(S) - tau, 0));    % Soft-thresholding over singular values
    % [U,S,V] = svds(X,1);                                  % Hard-thresholding (the matrix is known to have rank 1)
    X = U * S * V';

    if norm(Amap(X) - y) < 1e-3; fprintf("The method has converged in iteration %i!\n", i); break; end
    
end

[V,D] = eig(X); [~,i] = max(abs(diag(D))); recovered_x = V(:,i);
recovered_x = recovered_x * (recovered_x' * x) / abs(recovered_x' * x);
recovered_x = recovered_x * (norm(x) / norm(recovered_x));
original_x = reshape(x,n,n); recovered_x = reshape(recovered_x,n,n);

error = norm(recovered_x - original_x, 'fro') / norm(original_x, 'fro');
time = toc;

end
