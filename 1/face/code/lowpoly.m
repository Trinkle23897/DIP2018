function [] = lowpoly(I, fg_sat_boost, bg_hue_offset, fineness)
    m = size(I, 1);
    n = size(I, 2);
    [edge_map, foreground_mask] = segment(I);
%     sampled_points = sample([m, n], 'naive', 0.005);
%     sampled_points = sample([m, n], 'edge', 0.1, 0.001, edge_map);
    p1 = 0.2 * fineness;
    p21 = 0.001 * fineness;
    p22 = 0.0005 * fineness;
    sampled_points = sample([m, n], 'optim', p1, p21, p22, edge_map, foreground_mask, false);
    triangles = delaunay(sampled_points(:, 1), sampled_points(:, 2));
%     triplot(triangles, sampled_points(:, 2), sampled_points(:, 1));
    figure;
    imshow(I);
    hold on;
    I = rgb2hsv(im2uint8(I));
    sat_boosted = I(:, :, 2) * (1 + fg_sat_boost);
    hue_with_offset = mod(I(:, :, 1) + bg_hue_offset, 1);
    I(:, :, 2) = min(1, I(:, :, 2) .* ~foreground_mask + sat_boosted .* foreground_mask);
    I(:, :, 1) = I(:, :, 1) .* foreground_mask + hue_with_offset .* ~foreground_mask;
    I = hsv2rgb(I);
    for i = 1:size(triangles, 1)
        fprintf('Filling triangle %d/%d\n', i, size(triangles, 1));
        triangle = triangles(i, :);
%         roi_mask = poly2mask(sampled_points(triangle, 2), sampled_points(triangle, 1), m, n);
%         color = sample_color('rgb_mean', I, roi_mask);
        color = sample_color('com', I, [sampled_points(triangle, 1), sampled_points(triangle, 2)]);
        color = max(0, min(color, 1));
        patch(sampled_points(triangle, 2), sampled_points(triangle, 1), color, 'EdgeColor', 'none');
    end
    saveas(gcf, 'output.png');
end