function [mcoh,time,error,recovered_x] = phaselift_cvx(image,delta)

tic
n = size(image,1);
x = image(:);

masks = 4;
d = randi([0,1], masks, n^2);
F = dftmtx(n);
A = kron(F,F);
parfor i = 1:masks-1
    W = diag(d(i,:));
    A = [A; kron(F,F) * W];
end
m = size(A,1);
y = abs(A * x).^2;

A_newton = zeros(m,n^4);
parfor i = 1:m
    a = A(i,:);
    A_newton(i,:) = reshape(a'*a,n^4,1);
end

% Additive noise to intensity data
if nargin == 1; delta = 0; end
noise = delta * abs(y) .* randn(m,1);
y = y + noise;

% Mutual coherence of the measurement vectors
mcoh = abs(A' * A);
mcoh = mcoh - diag(diag(mcoh));

cvx_begin sdp
    cvx_precision high
    variable X(n^2,n^2) hermitian
    minimize trace(X)
    subject to
        X >= 0
        A_newton * X(:) == y
cvx_end

[V,D] = eig(X); [~,i] = max(abs(diag(D))); recovered_x = V(:,i);
recovered_x = recovered_x * (recovered_x' * x) / abs(recovered_x' * x);
recovered_x = recovered_x * (norm(x) / norm(recovered_x));
original_x = reshape(x,n,n); recovered_x = reshape(recovered_x,n,n);

error = norm(recovered_x - original_x, 'fro') / norm(original_x, 'fro');
time = toc;

end

