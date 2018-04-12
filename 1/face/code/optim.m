function points = optim(points, edge_map, foreground_mask, n_iter)
    % Lloyd relaxation with the constraint that foreground points are kept in foreground, background
    % points are kept in background, and points on edges are always on edges.
       
    fprintf('Optimizing %d points for %d iterations\n', size(points, 1), n_iter);
    
    [m, n] = size(foreground_mask);
    
    % label connected components
    label = bwlabel(foreground_mask);
    n_cc = max(max(label));
    region_masks = false(m, n, n_cc);
    for i = 1:n_cc
        region_masks(:, :, i) = label == i;
    end
    
    for counter = 1:n_iter
        
        fprintf('Optimizing: iteration %d/%d\n', counter, n_iter);
        
        % get voronoi vertices and cells
        rg = max(m, n);
        midx = round(m / 2);
        midy = round(n / 2);
        voronoin_points = [points; midx, midy - 5 * rg; midx, midy + 5 * rg; midx - 5 * rg, midy; midx + 5 * rg, midy];
        [V, C] = voronoin(voronoin_points);
        C = C(1:end - 4);
        
        for i = 1:length(C)
            % limit each voronoi cell to the connected component it belongs to
            current_cell = C{i};
            cell_poly = zeros(length(current_cell), 2);
            for j = 1:length(current_cell)
                cell_poly(j, :) = V(current_cell(j), :);
            end
            cell_mask = poly2mask(cell_poly(:, 2), cell_poly(:, 1), m, n);
            point = points(i, :);
            if ~foreground_mask(point(1), point(2))
                cell_mask = cell_mask & ~foreground_mask;
            else
                cell_mask = cell_mask & region_masks(:, :, label(point(1), point(2)));
            end
            
            % move the point to its new location
            stat = regionprops(cell_mask);
            if isempty(stat)
                continue;
            end
            centroid = stat.Centroid;
            centroid = round([centroid(2), centroid(1)]);
            if ~all(centroid > 0) || ~all(centroid <= [m, n])
                continue;
            end
            if edge_map(point)
                revised_point = find_nearest(edge_map, centroid);
                if all(revised_point)
                    points(i, :) = revised_point;
                end
            else
                if foreground_mask(centroid) == foreground_mask(point)
                    points(i, :) = centroid;
                end
            end
        end
    end
end

function nearest = find_nearest(mask, ref_point)
    nearest = [0, 0];
    x_ref = ref_point(1);
    y_ref = ref_point(2);
    r = 0;
    while x_ref - r >= 1 && x_ref + r <= m && y_ref - r >= 1 && y_ref + r <= n
        roi = mask(x_ref - r:x_ref + r, y_ref - r:y_ref + r);
        ind = find(roi, 1);
        if ~isempty(ind)
            [x_target, y_target] = ind2sub(ind);
            nearest = [x_ref + x_target - r - 1, y_ref + y_target - r - 1];
            break;
        end
        r = r + 1;
    end
end