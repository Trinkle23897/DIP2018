function [] = combine(I1, I2, bg_h, fg_sat_boost, fineness)
    m = 512;
    I1 = imresize(I1, [m, NaN]);
    I2 = imresize(I2, [m, NaN]);
    n1 = size(I1, 2);
    n2 = size(I2, 2);
    n = n1 + n2;
    
    % segment each image and combine the results
    [edge_map_1, foreground_mask_1] = segment(im2double(I1));
    [edge_map_2, foreground_mask_2] = segment(im2double(I2));
    edge_map = [edge_map_1, edge_map_2];
    foreground_mask = [foreground_mask_1, foreground_mask_2];
    
    % color shift in HSV space
    I1 = rgb2hsv(I1);
    I2 = rgb2hsv(I2);
    roi1 = I1(:, round(0.9 * size(I1, 2)):end, :);
    roi2 = I2(:, 1:round(0.1 * size(I2, 2)), :);
    h1 = get_h_center(roi1(:, :, 1));
    h2 = get_h_center(roi2(:, :, 1));
    mat_median = @(A) median(A(:));
    s1 = mat_median(roi1(:, :, 2));
    s2 = mat_median(roi2(:, :, 2));
    s = mat_median([roi1(:, :, 2), roi2(:, :, 2)]);
    v1 = mat_median(roi1(:, :, 3));
    v2 = mat_median(roi2(:, :, 3));
    v = mat_median([roi1(:, :, 3), roi2(:, :, 3)]);
    h_offset_1 = bg_h - h1;
    h_offset_2 = bg_h - h2;
    s_offset_1 = s - s1;
    s_offset_2 = s - s2;
    v_offset_1 = v - v1;
    v_offset_2 = v - v2;
    h_with_offset_1 = mod(I1(:, :, 1) + h_offset_1, 1);
    h_with_offset_2 = mod(I2(:, :, 1) + h_offset_2, 1);
    s_with_offset_1 = max(0, min(1, I1(:, :, 2) + s_offset_1));
    s_with_offset_2 = max(0, min(1, I2(:, :, 2) + s_offset_2));
    v_with_offset_1 = max(0, min(1, I1(:, :, 3) + v_offset_1));
    v_with_offset_2 = max(0, min(1, I2(:, :, 3) + v_offset_2));
    I1(:, :, 1) = I1(:, :, 1) .* foreground_mask_1 + h_with_offset_1 .* ~foreground_mask_1;
    I2(:, :, 1) = I2(:, :, 1) .* foreground_mask_2 + h_with_offset_2 .* ~foreground_mask_2;
    I1(:, :, 2) = I1(:, :, 2) .* foreground_mask_1 + s_with_offset_1 .* ~foreground_mask_1;
    I2(:, :, 2) = I2(:, :, 2) .* foreground_mask_2 + s_with_offset_2 .* ~foreground_mask_2;
    I1(:, :, 3) = I1(:, :, 3) .* foreground_mask_1 + v_with_offset_1 .* ~foreground_mask_1;
    I2(:, :, 3) = I2(:, :, 3) .* foreground_mask_2 + v_with_offset_2 .* ~foreground_mask_2;
    I = [I1, I2];
    sat_boosted = I(:, :, 2) * (1 + fg_sat_boost);
    I(:, :, 2) = min(1, I(:, :, 2) .* ~foreground_mask + sat_boosted .* foreground_mask);
    I = hsv2rgb(I);
    
    % sampling, triangulation and coloring
    p1 = 0.1 * fineness;
    p21 = 0.001 * fineness;
    p22 = 0.0005 * fineness;
    sampled_points = sample([m, n], 'optim', p1, p21, p22, edge_map, foreground_mask, false);
    triangles = delaunay(sampled_points(:, 1), sampled_points(:, 2));
    
    figure; imshow(I); hold on;
    for i = 1:size(triangles, 1)
        fprintf('Filling triangle %d/%d\n', i, size(triangles, 1));
        triangle = triangles(i, :);
        color = sample_color('com', I, [sampled_points(triangle, 1), sampled_points(triangle, 2)]);
        color = max(0, min(color, 1));
        patch(sampled_points(triangle, 2), sampled_points(triangle, 1), color, 'EdgeColor', 'none');
    end
end

function h = get_h_center(roi)
    [N, edges] = histcounts(roi);
    [~, argmax] = max(N);
    h = mean(edges(argmax:(argmax+1)));
    h = max(0, min(h, 1));
end