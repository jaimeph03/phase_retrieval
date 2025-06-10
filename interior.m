function [mcoh,time,error,recovered_x] = interior(image,delta)

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

A_newton = zeros(m,n^4);
parfor i = 1:m
    a = A(i,:);
    A_newton(i,:) = reshape(a'*a,n^4,1);
end

max_iter = 25;
t = 0.01;
X = eye(n^2);

for i = 1:max_iter
    
    for j = 1:15
        invX = X \ eye(n^2);
        grad = eye(n^2) - invX / t;
        hess = kron(invX,invX);
        res = zeros(m,1);
        parfor k = 1:m
            a = A(k,:);
            res(k) = a * X * a' - y(k);
        end
        mat = [hess A_newton'; A_newton zeros(m)];
        rhs = - [grad(:); res];

        aux = mat \ rhs;
        dX = reshape(aux(1:n^4),n^2,n^2);
        dX = (dX + dX') / 2;

        beta = 1;
        while beta > 1e-5
            if min(eig(X+beta*dX)) > 1e-8; break; end
            beta = beta * 0.5;
        end

        Xpre1 = X;
        X = X + beta * dX;

        if norm(X - Xpre1, "fro") < 1e-3; break; end

    end

    t = 2 * t;
    fprintf("Iteration %i complete!\n",i)
    if norm(A_newton * X(:) - y) < 1e-3; fprintf("The method has converged in iteration %i!\n", i); break; end

end

[V,D] = eig(X); [~,i] = max(abs(diag(D))); recovered_x = V(:,i);
recovered_x = recovered_x * (recovered_x' * x) / abs(recovered_x' * x);
recovered_x = recovered_x * (norm(x) / norm(recovered_x));
original_x = reshape(x,n,n); recovered_x = reshape(recovered_x,n,n);


error = norm(recovered_x - original_x, 'fro') / norm(original_x, 'fro');
time = toc;

end
