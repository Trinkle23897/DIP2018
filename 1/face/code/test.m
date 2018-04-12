I = imresize(im2double(imread('images/messi.jpg')), [512, NaN]);
lowpoly(I, 0, 0, 1);

% I1 = imread('images/cheng.jpg');
% I2 = imread('images/yun.jpg');
% combine(I1, I2, .5, .2, 1);