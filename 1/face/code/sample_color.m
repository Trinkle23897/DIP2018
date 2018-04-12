function color = sample_color(varargin)
    % Usage:
    % sample_color('rgb_mean', <RGB image>, <ROI mask>)
    % sample_color('com', <color image>, <vertices>);
    algorithm = varargin{1};
    I = varargin{2};
    [m, n, ~] = size(I);
    switch algorithm
        case 'rgb_mean'
            color = zeros(1, 3);
            roi_mask = varargin{3};
            for channel = 1:3
                I_channel = I(:, :, channel);
                color(channel) = mean(I_channel(roi_mask));
            end
        case 'com'
            vertices = varargin{3};
            com = round(mean(vertices, 1));
            com(1) = max(1, min(com(1), m));
            com(2) = max(1, min(com(2), n));
            color = reshape(I(com(1), com(2), :), [1, 3]);
    end
end