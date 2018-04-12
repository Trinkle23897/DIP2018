function [edge_map, foreground_mask] = segment(I)
    if ndims(I) == 3
        I = rgb2gray(I);
    end
    
    [m, n] = size(I);
    
    % basic edge detection using Canny algorithm
    canny_edge = edge(I, 'canny');
    
    % close small gaps
    closed_edge = imclose(canny_edge, strel('disk', 3));
    closed_edge = bwmorph(closed_edge, 'bridge', Inf);
    
    % get foreground by filling holes, pruning and removing small CCs
    closed_edge(m, :) = 1;
    filled = imfill(closed_edge, 'holes');
    pruned = prune(filled, 30);
    prune_opened = imopen(pruned, strel('disk', 1));
    foreground_mask = bwareaopen(prune_opened, 5000);
    
    edge_map = canny_edge & foreground_mask;
    
end