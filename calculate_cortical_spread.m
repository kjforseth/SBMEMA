% Summary
%   For each intracranial electrode, compute a weight
%   in [0 1] for every nearby cortical surface vertex. Weights equal 1 within
%   the contact radius and decay exponentially with Euclidean distance beyond
%   it; vertices are limited to a geodesic neighborhood along the cortical
%   surface so that activity does not "jump" across sulci or between banks.
%
%       w(v) = exp(-decay * max(0, ||x_v - x_e|| - contactRadius))
%
%   Two electrode models are supported:
%     'sde'  - subdural grids/strips. The contact footprint is the patch of
%              the smoothed outer pial envelope within contactRadius of the
%              contact (geodesic), projected onto the pial surface. Spread is
%              then restricted to pial vertices within spreadRadius
%              (geodesic) of that footprint.
%     'seeg' - depth electrodes. Each contact is classified as gray matter,
%              white matter, or outside cortex (CSF) from the nearest pial and
%              white matter vertices. Spread is restricted to pial vertices
%              within contactRadius + spreadRadius (geodesic) of:
%                gray:  the nearest pial vertex
%                white: pial vertices within contactRadius of the contact
%                CSF:   same as white, applied to every surface (hemisphere)
%
%   No lab-specific data structures or toolboxes are required. Geodesic
%   distances use perform_fast_marching_mesh() (Toolbox Fast Marching,
%   G. Peyre) when it is on the path, otherwise an exact-on-the-graph
%   Dijkstra along mesh edges (slight overestimate of true geodesic), or any
%   user supplied function.
%
% Input
%   coords: [Ne x 3] electrode coordinates (mm) in the same space as the
%           surfaces. Rows containing NaN are skipped (empty weights).
%   pial:   cortical surface mesh(es) to put weights on. Any of:
%             - a single mesh (see "Mesh formats" below)
%             - a cell array of meshes, e.g. {lh, rh}
%             - a struct of named meshes, e.g. struct('LH',lh,'RH',rh)
%   Name-value pairs (case insensitive):
%     'Type'          'sde' (default) | 'seeg'
%     'ContactRadius' radius (mm) of full weight around each contact
%                     (default 3)
%     'SpreadRadius'  geodesic extent (mm) of spread beyond the contact
%                     (default 7)
%     'Decay'         exponential decay constant (1/mm) (default 0.3)
%     'Threshold'     weights below this are dropped (default 1e-3)
%     'Envelope'      'sde' only: smoothed outer pial (e.g. FreeSurfer
%                     lh.pial-outer-smoothed), same layout as pial. If
%                     omitted, the footprint is the pial vertices within
%                     ContactRadius (Euclidean) of the contact.
%     'White'         'seeg' only (required): white matter mesh(es), same
%                     layout as pial.
%     'Hemisphere'    'sde' only: [Ne x 1] surface assignment for each
%                     electrode, as indices into pial or as names (cellstr)
%                     matching the fields of a struct pial. 0 or '' skips the
%                     electrode. If omitted, each electrode is assigned to the
%                     closest surface (envelope if given, else pial).
%     'Geodesic'      'auto' (default) | 'fastmarching' | 'dijkstra' |
%                     function handle d = fcn(vertices,faces,seeds,dmax)
%                     returning [Nv x 1] distances (Inf beyond dmax)
%     'Verbose'       true | false (default)
%
%   Mesh formats: a struct (or object, e.g. gifti) with vertex field
%   vertices | v | pos | vert and face field faces | f | tri, or a cell
%   {vertices, faces}. Vertices [Nv x 3]; faces [Nf x 3], 0- or 1-based.
%
% Output
%   W:    sparse [Nv x Ne] weight matrix per surface, in the same layout as
%         pial (matrix | cell of matrices | struct of matrices). Column e
%         holds the weights for electrode e.
%   info: struct with fields
%           surface   [Ne x 1] index of the surface each electrode was
%                     assigned to (0 = skipped)
%           tissue    [Ne x 1] cellstr: 'gray' | 'white' | 'csf' for 'seeg',
%                     'surface' for 'sde', '' if skipped
%           names     surface names
%           params    parameters used
%
% Example
%   Subdural grid on a left hemisphere:
%       W = calculate_cortical_spread(coords,lhPial,'Envelope',lhOuter);
%   Bilateral sEEG with larger contacts:
%       [W,info] = calculate_cortical_spread(coords,struct('LH',lh,'RH',rh),...
%           'Type','seeg','White',struct('LH',lhWhite,'RH',rhWhite),...
%           'ContactRadius',2);
%   Project electrode values onto the surface (weighted sum):
%       map = W.LH * values(:);
%   Coverage (maximum weight at each vertex):
%       coverage = full(max(W.LH,[],2));
%   Legacy get_cortical_spread() format (Z{e}.I, Z{e}.V):
%       [I,~,V] = find(W(:,e)); Z{e} = struct('I',I,'V',V);


function [W,info] = calculate_cortical_spread(coords,pial,varargin)

% handle inputs
p = inputParser;
p.FunctionName = 'calculate_cortical_spread';
p.addRequired('coords',@(x) isnumeric(x) && size(x,2) == 3);
p.addRequired('pial');
p.addParameter('Type','sde',@(x) any(strcmpi(x,{'sde' 'seeg'})));
p.addParameter('ContactRadius',3,@(x) isscalar(x) && x >= 0);
p.addParameter('SpreadRadius',7,@(x) isscalar(x) && x >= 0);
p.addParameter('Decay',0.3,@(x) isscalar(x) && x >= 0);
p.addParameter('Threshold',1e-3,@(x) isscalar(x) && x >= 0);
p.addParameter('Envelope',[]);
p.addParameter('White',[]);
p.addParameter('Hemisphere',[]);
p.addParameter('Geodesic','auto');
p.addParameter('Verbose',false,@(x) islogical(x) || isnumeric(x));
p.parse(coords,pial,varargin{:});
opt = p.Results;
opt.Type = lower(opt.Type);

coords = double(coords);
Ne = size(coords,1);

% parse surfaces
[P,names,layout] = parse_surfaces(pial,'pial');
Ns = length(P);
if ~isempty(opt.Envelope)
    E = parse_surfaces(opt.Envelope,'Envelope',names);
else
    E = {};
end
if ~isempty(opt.White)
    Wm = parse_surfaces(opt.White,'White',names);
else
    Wm = {};
end
if strcmp(opt.Type,'seeg') && isempty(Wm)
    error('calculate_cortical_spread: ''White'' surface(s) required for ''seeg''')
end

% geodesic distance function
geodesic = get_geodesic(opt.Geodesic,opt.Verbose);

% parameters
cR = opt.ContactRadius;
sR = opt.SpreadRadius;
decay = opt.Decay;
thresh = opt.Threshold;

% outputs
info.surface = zeros(Ne,1);
info.tissue  = repmat({''},Ne,1);
info.names   = names;
info.params  = struct('Type',opt.Type,'ContactRadius',cR,'SpreadRadius',sR,...
    'Decay',decay,'Threshold',thresh);
I = cell(Ns,Ne);
V = cell(Ns,Ne);

switch opt.Type
    case 'sde'
        % assign electrodes to surfaces
        if isempty(E), S = P; else S = E; end
        hemis = assign_surfaces(opt.Hemisphere,coords,S,names);

        for e = 1:Ne
            ecoord = coords(e,:);
            h = hemis(e);
            if any(isnan(ecoord)) || (h == 0), continue, end

            % contact footprint on pial
            if ~isempty(E)
                [~,seedE] = min(euclidean(E{h}.v,ecoord));
                dist = geodesic(E{h},seedE,cR);
                Inz = find(dist <= cR);
                seeds = zeros(length(Inz),1);
                for n = 1:length(Inz)
                    [~,seeds(n)] = min(euclidean(P{h}.v,E{h}.v(Inz(n),:)));
                end
                seeds = unique(seeds);
            else
                seeds = nearby(P{h}.v,ecoord,cR);
            end

            % spread from footprint
            dist = geodesic(P{h},seeds,sR);
            [I{h,e},V{h,e}] = spread(P{h}.v,ecoord,dist,sR,cR,decay,thresh);

            info.surface(e) = h;
            info.tissue{e}  = 'surface';
            if opt.Verbose, fprintf('Electrode %d: %s\n',e,names{h}), end
        end

    case 'seeg'
        for e = 1:Ne
            ecoord = coords(e,:);
            if any(isnan(ecoord)), continue, end

            % distances to every pial & white vertex
            dP = cell(Ns,1); dW = cell(Ns,1);
            aP = zeros(Ns,2); aW = zeros(Ns,2);
            for h = 1:Ns
                dP{h} = euclidean(P{h}.v,ecoord);
                dW{h} = euclidean(Wm{h}.v,ecoord);
                [aP(h,1),aP(h,2)] = min(dP{h});
                [aW(h,1),aW(h,2)] = min(dW{h});
            end

            % anchor with closest point, then pair pial & white vertices
            [~,a] = min([aP(:,1); aW(:,1)]);
            if a <= Ns
                h  = a;
                Ig = aP(h,2);
                [~,Iw] = min(euclidean(Wm{h}.v,P{h}.v(Ig,:)));
            else
                h  = a - Ns;
                Iw = aW(h,2);
                [~,Ig] = min(euclidean(P{h}.v,Wm{h}.v(Iw,:)));
            end

            % project contact onto white -> pial axis
            g = P{h}.v(Ig,:);
            w = Wm{h}.v(Iw,:);
            Lwg = g - w;
            if norm(Lwg) == 0
                tissue = 'gray';
            else
                u = Lwg/norm(Lwg);
                Vg = sum((g - ecoord).*u);
                Vw = sum((w - ecoord).*u);
                if Vg*Vw < 0
                    tissue = 'gray';
                elseif abs(Vg) > abs(Vw)
                    tissue = 'white';
                else
                    tissue = 'csf';
                end
            end

            dmax = cR + sR;
            switch tissue
                case 'gray'
                    dist = geodesic(P{h},Ig,dmax);
                    [I{h,e},V{h,e}] = spread(P{h}.v,ecoord,dist,dmax,cR,decay,thresh);
                case 'white'
                    seeds = find(dP{h} < cR);
                    if isempty(seeds), seeds = Ig; end
                    dist = geodesic(P{h},seeds,dmax);
                    [I{h,e},V{h,e}] = spread(P{h}.v,ecoord,dist,dmax,cR,decay,thresh);
                case 'csf'
                    for hn = 1:Ns
                        seeds = find(dP{hn} < cR);
                        if isempty(seeds), seeds = aP(hn,2); end
                        dist = geodesic(P{hn},seeds,dmax);
                        [I{hn,e},V{hn,e}] = spread(P{hn}.v,ecoord,dist,dmax,cR,decay,thresh);
                    end
            end

            info.surface(e) = h;
            info.tissue{e}  = tissue;
            if opt.Verbose, fprintf('Electrode %d: %s %s\n',e,names{h},tissue), end
        end
end

% assemble sparse weight matrices
Wc = cell(Ns,1);
for h = 1:Ns
    Nv = size(P{h}.v,1);
    rows = vertcat(I{h,:});
    cols = cell2mat(arrayfun(@(e) repmat(e,numel(I{h,e}),1),(1:Ne)','UniformOutput',false));
    vals = vertcat(V{h,:});
    if isempty(rows), rows = zeros(0,1); cols = zeros(0,1); vals = zeros(0,1); end
    Wc{h} = sparse(rows,cols,vals,Nv,Ne);
end

% handle output in the same layout as pial
switch layout
    case 'single', W = Wc{1};
    case 'cell',   W = reshape(Wc,size(pial));
    case 'struct', W = cell2struct(Wc,names,1);
end



%% helpers
function d = euclidean(v,x)
% distance from every row of v to point x
d = sqrt(sum(bsxfun(@minus,v,x).^2,2));


function I = nearby(v,x,r)
% vertices within r of x, else the nearest vertex
d = euclidean(v,x);
I = find(d <= r);
if isempty(I), [~,I] = min(d); end


function [I,V] = spread(v,ecoord,dist,dmax,cR,decay,thresh)
% weights for vertices within geodesic dmax
I = find(~isinf(dist) & (dist <= dmax));
V = exp(-decay*max(0,euclidean(v(I,:),ecoord) - cR));
del = (V < thresh);
I(del) = []; V(del) = [];


function hemis = assign_surfaces(hemi,coords,S,names)
% surface index for each electrode
Ne = size(coords,1);
Ns = length(S);
if isempty(hemi)
    if Ns == 1
        hemis = ones(Ne,1);
    else
        d = inf(Ne,Ns);
        for h = 1:Ns
            for e = 1:Ne
                d(e,h) = min(euclidean(S{h}.v,coords(e,:)));
            end
        end
        [~,hemis] = min(d,[],2);
    end
elseif isnumeric(hemi) || islogical(hemi)
    hemis = double(hemi(:));
    if isscalar(hemis), hemis = repmat(hemis,Ne,1); end
else
    if ischar(hemi), hemi = cellstr(hemi); end
    if isstring(hemi), hemi = cellstr(hemi); end
    if isscalar(hemi), hemi = repmat(hemi,Ne,1); end
    hemis = zeros(numel(hemi),1);
    for e = 1:numel(hemi)
        if isempty(hemi{e}), continue, end
        h = find(strcmpi(strtrim(hemi{e}),names));
        if isempty(h), error('calculate_cortical_spread: unknown Hemisphere ''%s''',hemi{e}), end
        hemis(e) = h;
    end
end
if numel(hemis) ~= Ne
    error('calculate_cortical_spread: Hemisphere must have one entry per electrode')
end
if any(hemis < 0 | hemis > Ns | hemis ~= round(hemis))
    error('calculate_cortical_spread: Hemisphere indices must be in 0:%d',Ns)
end


function [M,names,layout] = parse_surfaces(s,label,names)
% normalize single | cell | named struct input to a cell of meshes
if iscell(s)
    layout = 'cell';
    M = cellfun(@(x) parse_mesh(x,label),s(:),'UniformOutput',false);
    if nargin < 3, names = arrayfun(@(n) sprintf('%d',n),(1:numel(s))','UniformOutput',false); end
elseif isstruct(s) && isscalar(s) && isempty(mesh_field(s,'v'))
    layout = 'struct';
    fn = fieldnames(s);
    if nargin == 3
        if ~all(ismember(names,fn)), error('calculate_cortical_spread: %s must have fields %s',label,strjoin(names,', ')), end
        fn = names;
    end
    M = cellfun(@(f) parse_mesh(s.(f),label),fn,'UniformOutput',false);
    names = fn;
else
    layout = 'single';
    M = {parse_mesh(s,label)};
    if nargin < 3, names = {'1'}; end
end
if nargin == 3 && numel(M) ~= numel(names)
    error('calculate_cortical_spread: %s must have the same number of surfaces as pial',label)
end


function m = parse_mesh(s,label)
% extract vertices & faces from common mesh formats
if iscell(s) && numel(s) == 2
    v = s{1}; f = s{2};
else
    vf = mesh_field(s,'v'); ff = mesh_field(s,'f');
    if isempty(vf) || isempty(ff)
        error('calculate_cortical_spread: could not find vertices/faces in %s',label)
    end
    v = s.(vf); f = s.(ff);
end
v = double(v); f = double(f);
if size(v,2) ~= 3 && size(v,1) == 3, v = v'; end
if size(f,2) ~= 3 && size(f,1) == 3, f = f'; end
if min(f(:)) == 0, f = f + 1; end
if size(v,2) ~= 3 || size(f,2) ~= 3 || max(f(:)) > size(v,1)
    error('calculate_cortical_spread: invalid mesh in %s',label)
end
m = struct('v',v,'f',f);


function name = mesh_field(s,type)
% name of the vertex or face field in s, '' if none
switch type
    case 'v', candidates = {'vertices' 'v' 'pos' 'vert'};
    case 'f', candidates = {'faces' 'f' 'tri'};
end
name = '';
if isstruct(s)
    fn = fieldnames(s);
elseif isobject(s)
    try fn = fieldnames(s); catch, fn = {}; end
    if isempty(fn), try fn = properties(s); catch, fn = {}; end, end
    if isa(s,'gifti'), fn = [fn(:); {'vertices'; 'faces'}]; end
else
    return
end
I = find(ismember(candidates,fn),1);
if ~isempty(I), name = candidates{I}; end


function geodesic = get_geodesic(method,verbose)
% returns geodesic = @(mesh,seeds,dmax) -> [Nv x 1] distances
if isa(method,'function_handle')
    geodesic = @(m,seeds,dmax) reshape(method(m.v,m.f,seeds,dmax),[],1);
    return
end
switch lower(method)
    case 'auto'
        if exist('perform_fast_marching_mesh','file')
            method = 'fastmarching';
        else
            method = 'dijkstra';
        end
        if verbose, fprintf('Geodesic distance: %s\n',method), end
    case {'fastmarching' 'dijkstra'}
        if strcmpi(method,'fastmarching') && ~exist('perform_fast_marching_mesh','file')
            error('calculate_cortical_spread: perform_fast_marching_mesh not on path (Toolbox Fast Marching)')
        end
    otherwise
        error('calculate_cortical_spread: unknown Geodesic method ''%s''',method)
end
switch lower(method)
    case 'fastmarching'
        geodesic = @(m,seeds,dmax) reshape(perform_fast_marching_mesh(m.v,m.f,seeds(:),struct('dmax',dmax)),[],1);
    case 'dijkstra'
        geodesic = @geodesic_dijkstra;
end


function d = geodesic_dijkstra(m,seeds,dmax)
% shortest paths along mesh edges, limited to dmax
% (any path shorter than dmax stays within a Euclidean ball of dmax around
%  its seed, so only those vertices need to be included in the graph)
Nv = size(m.v,1);
near = false(Nv,1);
for s = 1:numel(seeds)
    near = near | (euclidean(m.v,m.v(seeds(s),:)) <= dmax);
end
f = m.f(any(near(m.f),2),:);
edges = unique(sort([f(:,[1 2]); f(:,[2 3]); f(:,[3 1])],2),'rows');
edges = edges(all(near(edges),2),:);
local = find(near);
map = zeros(Nv,1); map(local) = 1:numel(local);
len = sqrt(sum((m.v(edges(:,1),:) - m.v(edges(:,2),:)).^2,2));
G = graph(map(edges(:,1)),map(edges(:,2)),len,numel(local));
D = min(distances(G,map(seeds(:))),[],1);
d = inf(Nv,1);
d(local) = D;
d(d > dmax) = inf;
