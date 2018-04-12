function points = sample(varargin)
    % Usage:
    % sample(<size of image>, 'naive', <probability for selection>)
    % sample(<size of image>, 'edge', <probability on edge>, <probability off edge>, <edge map>)
    % sample(<size of image>, 'nonuniform', <probability on edge>, <probability for
    %   foreground>, <probability for background>, <edge map>, <foreground mask>);
    % sample(<size of image>, 'optim', <probability on edge>, <probability for foreground>,
    %   <probability for background>, <edge map>, <foreground mask>, visualize);
    sz = varargin{1};
    m = sz(1);
    n = sz(2);
    algorithm = varargin{2};
    sampled_mask = false(m, n);
    switch algorithm
        case 'naive'
            p = varargin{3};
            for i = 1:m
                for j = 1:n
                    sampled_mask(i, j) = rand < p;
                end
            end
        case 'edge'
            p1 = varargin{3};
            p2 = varargin{4};
            edge_map = varargin{5};
            for i = 1:m
                for j = 1:n
                    sampled_mask(i, j) = (edge_map(i, j) && rand < p1) || (~edge_map(i, j) && rand < p2);
                end
            end
        case {'nonuniform', 'optim'}
            p1 = varargin{3};
            p21 = varargin{4};
            p22 = varargin{5};
            edge_map = varargin{6};
            foreground_mask = varargin{7};
            for i = 1:m
                for j = 1:n
                    if edge_map(i, j)
                        sampled_mask(i, j) = rand < p1;
                    elseif foreground_mask(i, j)
                        sampled_mask(i, j) = rand < p21;
                    else
                        sampled_mask(i, j) = rand < p22;
                    end
                end
            end
    end
    
    if strcmp(algorithm, 'optim')
        visualize = varargin{8};
        [row, col] = find(sampled_mask);
        points = [row, col];
        if visualize
            triangles = delaunay(points(:, 1), points(:, 2));
            figure; hold on;
            subplot(1, 2, 1);
            triplot(triangles, points(:, 2), points(:, 1));
            set(gca, 'Ydir', 'reverse');
        end
        points = optim(points, edge_map, foreground_mask, 10);
        points = [points; 1, 1; m, 1; 1, n; m, n];
        if visualize
            triangles = delaunay(points(:, 1), points(:, 2));
            subplot(1, 2, 2);
            triplot(triangles, points(:, 2), points(:, 1));
            set(gca, 'Ydir', 'reverse');
            hold off;
        end
    else
        sampled_mask(1, 1) = 1;
        sampled_mask(m, 1) = 1;
        sampled_mask(1, n) = 1;
        sampled_mask(m, n) = 1;
        [row, col] = find(sampled_mask);
        points = [row, col];
    end
end