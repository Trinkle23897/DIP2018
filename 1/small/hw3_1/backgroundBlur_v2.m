I = imread('bobby.bmp');
I = im2double(I);
load('is_foreground.mat');
load('filter_size.mat');

m = size(I, 1);
n = size(I, 2);

I1 = applyMeanFilter(I, filter_size);
imshow(I1);
imwrite(I1, 'v2.bmp');

function I1 = applyMeanFilter(I, filter_size)
    % apply mean filter to I with same padding
    m = size(I, 1);
    n = size(I, 2);
    pad_size = (max(max(filter_size))-1) / 2;
    pad_topleft = I(1, 1) * ones(pad_size, pad_size);
    pad_top = repmat(I(1, :), pad_size, 1);
    pad_topright = I(1, n) * ones(pad_size, pad_size);
    pad_left = repmat(I(:, 1), 1, pad_size);
    pad_right = repmat(I(:, n), 1, pad_size);
    pad_bottomleft = I(m, 1) * ones(pad_size, pad_size);
    pad_bottom = repmat(I(m, :), pad_size, 1);
    pad_bottomright = I(m, n) * ones(pad_size, pad_size);
    I_padded = [
            pad_topleft,    pad_top,    pad_topright;
            pad_left,       I,          pad_right;
            pad_bottomleft, pad_bottom, pad_bottomright
        ];
    I1 = zeros(m, n); 
    for i = 1:m
        for j = 1:n
            current_filter_size = filter_size(i, j);
            half_size = (current_filter_size - 1)/2;
            filter = 1/current_filter_size^2 * ones(current_filter_size, current_filter_size);
            I1(i, j) = sum(sum(filter .* I_padded(...
                i+pad_size-half_size:i+pad_size+half_size, j+pad_size-half_size:j+pad_size+half_size...
            )));
        end
    end
end