clear

%this is the fft of a dirac
b=  zeros(32,32,32);
b(17,17,17) = 1;
bfft = fftn(b);

f = randn(32,32,32);
res = ifftn(fftn(f).*bfft);
norm(res(:)-f(:))



f =zeros(32,32,32);
f(1:4,1:4,1:4) = randn(4,4,4);
imagesc(f(:,:,1))

res = ifftshift(ifftn(fftn(f).*bfft));

 figure
imagesc(res(:,:,1))
title('result')