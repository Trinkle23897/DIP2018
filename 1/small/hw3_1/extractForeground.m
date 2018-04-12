I = imread('bobby_annotated.bmp');
m = size(I, 1);
n = size(I, 2);

figure; hold on;
subplot(2, 2, 1), imshow(I);

% extract contours
is_contour = false(m, n);
neighbors = [-1, -1; -1, 0; -1, 1; 0, -1; 0, 1; 1, -1; 1, 0; 1, 1];
anchors = [389, 425; 843, 416; 861, 801];   % one manually annotated point in each contour
for i_contour = 1:size(anchors, 1)
    is_contour(anchors(i_contour, 1), anchors(i_contour, 2)) = 1;
    % perform BFS starting from the anchor
    open_table = zeros(m * n, 2);
    open_table_start = 1;
    open_table(1, :) = anchors(i_contour, :);
    open_table_end = 1;
    while open_table_start <= open_table_end
        current_pixel = open_table(open_table_start, :);
        open_table_start = open_table_start + 1;
        for j = 1:size(neighbors, 1)
            neighbor_pixel = current_pixel + neighbors(j, :);
            if neighbor_pixel(1) >= 1 && neighbor_pixel(1) <= m && ...
                    neighbor_pixel(2) >= 1 && neighbor_pixel(2) <= n && ...
                    ~is_contour(neighbor_pixel(1), neighbor_pixel(2)) && ...
                    is_pixel_on_contour(I, neighbor_pixel)
                is_contour(neighbor_pixel(1), neighbor_pixel(2)) = 1;
                open_table_end = open_table_end + 1;
                open_table(open_table_end, :) = neighbor_pixel;
            end
        end
    end
end

subplot(2, 2, 2), imshow(is_contour);

% extract foreground
is_foreground = false(m, n);
neighbors = [-1, 0; 0, -1; 0, 1; 1, 0];     % only consider horizontal/vertical neighbors
anchors = [506, 557; 870, 446; 892, 867];   % one manually annotated point in each foreground area
for i_foreground = 1:size(anchors, 1)
    is_foreground(anchors(i_foreground, 1), anchors(i_foreground, 2)) = 1;
    % perform BFS flood filling starting from the anchor
    open_table = zeros(m * n, 2);
    open_table_start = 1;
    open_table(1, :) = anchors(i_foreground, :);
    open_table_end = 1;
    while open_table_start <= open_table_end
        current_pixel = open_table(open_table_start, :);
        open_table_start = open_table_start + 1;
        for j = 1:size(neighbors, 1)
            neighbor_pixel = current_pixel + neighbors(j, :);
            if neighbor_pixel(1) >= 1 && neighbor_pixel(1) <= m && ...
                    neighbor_pixel(2) >= 1 && neighbor_pixel(2) <= n && ...
                    ~is_contour(neighbor_pixel(1), neighbor_pixel(2)) && ...
                    ~is_foreground(neighbor_pixel(1), neighbor_pixel(2))
                is_foreground(neighbor_pixel(1), neighbor_pixel(2)) = 1;
                open_table_end = open_table_end + 1;
                open_table(open_table_end, :) = neighbor_pixel;
            end
        end
    end
end

is_foreground(is_contour == 1) = 1;
subplot(2, 2, 3), imshow(is_foreground);
save('is_foreground.mat', 'is_foreground');

function result = is_pixel_on_contour(I, coord)
    % check whether the pixel is red enough to check whether is belongs to
    % the manually annotated contour
    rgb = I(coord(1), coord(2), :);
    result = (rgb(1) > rgb(2)/2 + rgb(3)/2 + 5);
end
