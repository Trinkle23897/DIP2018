figure(); hold on;

for i = 1:4
    I = imread(sprintf('pollen%d.tif', i));
    [I1, hist_orig, hist_eq] = myHistEq(I);
    
    subplot(3, 4, i);
    imshow(I1);
    title(sprintf('Equalized image %d', i));
    
    subplot(3, 4, i+4);
    bar(hist_orig);
    axis([0, 255, 0, inf]);
    title(sprintf('Original histogram %d', i));
    
    subplot(3, 4, i+8);
    bar(hist_eq);
    axis([0, 255, 0, inf]);
    title(sprintf('Equalized histogram %d', i));
    
    imwrite(I1, sprintf('pollen%d_eq.tif', i));
end

function [I1, hist_original, hist_equalized] = myHistEq(I)
    % assume the range of I is 0~255
    hist_original = zeros(1, 256);
    for i = 0:255
        hist_original(i+1) = sum(sum(I == i));
    end
    p = hist_original / numel(I);
    s = 255 * cumsum(p);
    I1 = zeros(size(I, 1), size(I, 2));
    for i = 0:255
        I1(I == i) = s(i + 1);
    end
    I1 = uint8(I1);
    
    hist_equalized = zeros(1, 256);
    for i = 0:255
        hist_equalized(i+1) = sum(sum(I1 == i));
    end
end