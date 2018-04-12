I = imread('bobby.bmp');
I = im2double(I);
load('is_foreground.mat');

filter_sizes = [3, 5, 7, 9, 15, 25];

figure; hold on;

for i = 1:6
    filter_size = filter_sizes(i);
    filter = 1/filter_size^2 * ones(filter_size, filter_size);
    I1 = applyFilterToBackground(I, filter, is_foreground);
    subplot(2, 3, i);
    imshow(I1);
    title(sprintf('%dx%d', filter_size, filter_size));
    imwrite(I1, sprintf('v1_%dx%d.bmp', filter_size, filter_size));
end

function I1 = applyFilterToBackground(I, filter, is_foreground)
    % apply the filter to I by correlation, ignoring border    
    half_size = (size(filter, 1) - 1) / 2;
    I1 = I;
    for i = (1+half_size):(size(I, 1)-half_size)
        for j = (1+half_size):(size(I, 2)-half_size)
            if is_foreground(i, j)
                I1(i, j) = I(i, j);
            else
                I1(i, j) = sum(sum(filter .* I(i-half_size:i+half_size, j-half_size:j+half_size)));
            end
        end
    end
end