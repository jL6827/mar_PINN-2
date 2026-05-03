function drawZoom_final(axZ, Xq, Yq, Zq, xKm, yKm, z, lon0, lat0, km_per_deg_lon, km_per_deg_lat, viewPreset, centerLonLat, winDeg, idx, mainAx, zGlobalRange)
% drawZoom_final  Ultra-smooth anti-aliased zoom window

cLon = centerLonLat(1);
cLat = centerLonLat(2);

lonMin = cLon - winDeg(1)/2; lonMax = cLon + winDeg(1)/2;
latMin = cLat - winDeg(2)/2; latMax = cLat + winDeg(2)/2;

xMin = (lonMin - lon0) * km_per_deg_lon; xMax = (lonMax - lon0) * km_per_deg_lon;
yMin = (latMin - lat0) * km_per_deg_lat; yMax = (latMax - lat0) * km_per_deg_lat;

% 裁剪数据
mask = Xq >= xMin & Xq <= xMax & Yq >= yMin & Yq <= yMax;

Zq_masked = Zq;
Zq_masked(~mask) = NaN;
validPts = sum(~isnan(Zq_masked(:)));

if validPts < 10
    cla(axZ);
    text(axZ, 0.5, 0.5, sprintf('Insufficient data\nZoom %d', idx), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Units', 'normalized');
    axis(axZ, [0 1 0 1]);
    axZ.Visible = 'off';
    return;
end

% === 局部超采样插值 ===
% 扩大一点范围避免边缘效应
margin = 0.02;
xMinExt = xMin - margin*(xMax-xMin);
xMaxExt = xMax + margin*(xMax-xMin);
yMinExt = yMin - margin*(yMax-yMin);
yMaxExt = yMax + margin*(yMax-yMin);

% 使用更密的网格
nxLocal = 500;
nyLocal = 500;

xLocal = linspace(xMinExt, xMaxExt, nxLocal);
yLocal = linspace(yMinExt, yMaxExt, nyLocal);
[Xlocal, Ylocal] = meshgrid(xLocal, yLocal);

% 插值
F_local = scatteredInterpolant(xKm, yKm, z, 'natural', 'none');
Zlocal = F_local(Xlocal, Ylocal);

% === 多层平滑处理（不需要任何工具箱）===
% 1. 移动中值（替代medfilt2）
Zlocal = movmedian(Zlocal, 3, 1, 'omitnan');
Zlocal = movmedian(Zlocal, 3, 2, 'omitnan');

% 2. 高斯平滑
sigma = 2.5;  % 子图高斯平滑参数
kernelSize = ceil(6*sigma);
if mod(kernelSize,2)==0, kernelSize = kernelSize+1; end
x = -(kernelSize-1)/2:(kernelSize-1)/2;
[Xk, Yk] = meshgrid(x, x);
gaussKernel = exp(-(Xk.^2 + Yk.^2)/(2*sigma^2));
gaussKernel = gaussKernel / sum(gaussKernel(:));

Zlocal = conv2(Zlocal, gaussKernel, 'same');

% 3. 移动平均
Zlocal = smoothdata(Zlocal, 1, 'movmean', 5, 'omitnan');
Zlocal = smoothdata(Zlocal, 2, 'movmean', 5, 'omitnan');

% 裁剪回原始范围
maskExt = Xlocal >= xMin & Xlocal <= xMax & Ylocal >= yMin & Ylocal <= yMax;
Zlocal(~maskExt) = NaN;

% 提取显示区域
[rowMask, colMask] = find(maskExt);
if isempty(rowMask)
    cla(axZ);
    text(axZ, 0.5, 0.5, sprintf('No data in zoom area %d', idx), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Units', 'normalized');
    return;
end

rowRange = min(rowMask):max(rowMask);
colRange = min(colMask):max(colMask);

Xlocal_display = Xlocal(rowRange, colRange);
Ylocal_display = Ylocal(rowRange, colRange);
Zlocal_display = Zlocal(rowRange, colRange);

% === 渲染设置 ===
set(axZ.Parent, 'Renderer', 'opengl');

try
    set(axZ.Parent, 'GraphicsSmoothing', 'on');
catch
end

% 绘制曲面
hs = surf(axZ, Xlocal_display, Ylocal_display, Zlocal_display, ...
    'FaceColor', 'interp', ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 1.0, ...
    'FaceLighting', 'gouraud', ...
    'LineStyle', 'none');

% === 颜色和光照（与主图一致）===
if exist('mainAx', 'var') && ~isempty(mainAx)
    colormap(axZ, colormap(mainAx));
    clim(axZ, clim(mainAx));
else
    colormap(axZ, jet(256));
    if exist('zGlobalRange', 'var') && ~isempty(zGlobalRange)
        clim(axZ, zGlobalRange);
    end
end

axZ.ZDir = 'reverse';
camproj(axZ, 'orthographic');

% 宽高比
xRange = xMax - xMin;
yRange = yMax - yMin;
zRange = max(Zlocal_display(:)) - min(Zlocal_display(:));
daspect(axZ, [xRange/50, yRange/50, zRange/20]);

% view
switch viewPreset
    case "A"
        view(axZ, -135, 30);
    case "B"
        view(axZ, -135, 60);
    otherwise
        view(axZ, -135, 30);
end

% 光照（与主图相同）
delete(findall(axZ, 'Type', 'light'));

material(axZ, 'dull');
lighting(axZ, 'gouraud');
axZ.AmbientLightColor = [0.35 0.35 0.35];
camlight(axZ, 'headlight');
camlight(axZ, 45, 30);

hs.SpecularStrength = 0.15;
hs.DiffuseStrength = 0.85;
hs.AmbientStrength = 0.35;

% 设置范围
xlim(axZ, [xMin xMax]);
ylim(axZ, [yMin yMax]);

zValid = Zlocal_display(~isnan(Zlocal_display));
if ~isempty(zValid)
    zlim(axZ, [min(zValid), max(zValid)]);
end

% 精细刻度
nTicks = 4;
lonTicks = linspace(lonMin, lonMax, nTicks);
latTicks = linspace(latMin, latMax, nTicks);

axZ.XTick = (lonTicks - lon0) * km_per_deg_lon;
axZ.YTick = (latTicks - lat0) * km_per_deg_lat;
axZ.XTickLabel = compose('%.3f', lonTicks);
axZ.YTickLabel = compose('%.3f', latTicks);

% 美观设置
axZ.XGrid = 'on'; axZ.YGrid = 'on'; axZ.ZGrid = 'on';
axZ.GridAlpha = 0.15;
axZ.Box = 'on';
axZ.FontName = 'Arial';
axZ.LineWidth = 1.0;
axZ.FontSize = 9;
axZ.SortMethod = 'childorder';

title(axZ, sprintf('Zoom %d (%.3f, %.3f)', idx, cLon, cLat), ...
    'FontSize', 10, 'FontWeight', 'normal');
xlabel(axZ, 'Longitude', 'FontSize', 9);
ylabel(axZ, 'Latitude',  'FontSize', 9);
end