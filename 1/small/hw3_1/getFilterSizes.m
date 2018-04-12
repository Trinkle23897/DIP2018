load('is_foreground.mat');

m = size(is_foreground, 1);
n = size(is_foreground, 2);

% 1. determine filter sizes for background
MAX_FILTER_SIZE = 9;
THRESH = 330;    % pixels with x <= THRESH are considered distant; near otherwise
filter_size_vline = zeros(m, 1);    % consider a single vertical line first
filter_size_vline(1:THRESH) = MAX_FILTER_SIZE;
for i = 1:(m-THRESH)
    % linear interpolation for near objects
    filter_size_vline(THRESH+i) = MAX_FILTER_SIZE + (1-MAX_FILTER_SIZE) * i / (m-THRESH);
end
filter_size = repmat(filter_size_vline, 1, n);

% 2. disable filtering for foreground
filter_size(is_foreground) = 1;

% 3. smooth boundaries of foreground
filter_size_smooth = filter_size;
for i = 1:m
    for j = 1:n
        if is_boundary(is_foreground, i, j) &&...
                i >= 2 && i<= m-1 && j>=2 && j<= n-1
            filter_size_smooth(i, j) = 1/9 * sum(sum(...
                filter_size(i-1:i+1, j-1:j+1)...
            ));
        end
    end
end

roundToNearestOdd = @(A) 1 + 2 * round((A-1)/2);
filter_size = roundToNearestOdd(filter_size_smooth);

save('filter_size.mat', 'filter_size');

function result = is_boundary(is_foreground, i, j)
    m = size(is_foreground, 1);
    n = size(is_foreground, 2);
    if i <= 1 || i >= m || j <= 1 || j >= n
        result = false;
    else
        neighbor_sum = sum(is_foreground(i-1:i+1, j)) + sum(is_foreground(i, j-1:j+1)) - is_foreground(i, j);
        if neighbor_sum == 0 || neighbor_sum == 5
            % the neighbor fully belongs to foreground or background
            result = false;
        else
            result = true;
        end
    end
end