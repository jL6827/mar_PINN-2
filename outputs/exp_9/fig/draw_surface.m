% Thermocline depth 3D surface (scattered interpolation + aspect fix + lon/lat ticks + lighting + hi-res)
% Paper-ready version with aligned zoom windows and anti-aliasing

clear; clc;

%% 参数区
nx = 600;  % 提高到600，减少锯齿
ny = 600;

interpMethod = 'natural';   
extrapMethod = 'none';      
doSmooth = true;
smoothWin = 7;

viewPreset = "A";           

outname = 'Fig_ThermoclineSurface_StudyRegion_3D_withInsets_600dpi.png';

nLonTicks = 6;
nLatTicks = 6;

exportDPI = 600;

%% 1) 导入数据
data = load('thermocline_grid_data.mat');

Xg = data.Xg;
Yg = data.Yg;
Zg = data.Zg;

x_label   = data.x_label;   
y_label   = data.y_label;   
z_label   = data.z_label;   

%% 2) 散点化 + 去除 NaN/Inf
lon = Xg(:);
lat = Yg(:);
z   = Zg(:);

valid = isfinite(lon) & isfinite(lat) & isfinite(z);
lon = lon(valid); lat = lat(valid); z = z(valid);

[ll_unique, ia] = unique([lon lat], 'rows', 'stable');
lon = ll_unique(:,1);
lat = ll_unique(:,2);
z   = z(ia);

%% 3) 经纬度 -> 近似 km 坐标
lon0 = mean(lon, 'omitnan');
lat0 = mean(lat, 'omitnan');

km_per_deg_lat = 111.32;
km_per_deg_lon = 111.32 * cosd(lat0);

xKm = (lon - lon0) * km_per_deg_lon;   
yKm = (lat - lat0) * km_per_deg_lat;   

%% 4) 插值器
F = scatteredInterpolant(xKm, yKm, z, interpMethod, extrapMethod);

%% 5) 密网格 + 插值（更高分辨率减少锯齿）
xq = linspace(min(xKm), max(xKm), nx);
yq = linspace(min(yKm), max(yKm), ny);
[Xq, Yq] = meshgrid(xq, yq);

Zq = F(Xq, Yq);

if doSmooth
    Zq = smoothdata(Zq, 1, 'movmean', smoothWin, 'omitnan');
    Zq = smoothdata(Zq, 2, 'movmean', smoothWin, 'omitnan');
end

%% 6) 论文排版式布局 - colorbar在左边
fig = figure('Position', [80, 60, 1500, 800], 'Color', 'w');

% === 全局抗锯齿和渲染设置 ===
% 设置渲染器为OpenGL（硬件加速）
set(fig, 'Renderer', 'opengl');

% 启用图形平滑（MATLAB 2014b+）
try
    set(fig, 'GraphicsSmoothing', 'on');
catch
end

% 如果支持，设置OpenGL为硬件模式
try
    opengl('hardware');
catch
end

% 设置抗锯齿级别（某些版本支持）
try
    set(fig, 'DefaultSurfaceEdgeColor', 'none');
    set(fig, 'DefaultPatchEdgeColor', 'none');
catch
end

% 修改布局：colorbar在左边，主图在中间，子图在右边
% colorbar位置 [left bottom width height]
cbarPos = [0.03, 0.35, 0.015, 0.3];

% 主图位置（向右移动，给左边colorbar留空间）
mainPos = [0.10, 0.12, 0.60, 0.82];

% 子图位置（高度等于主图高度的一半）
subHeight = mainPos(4) / 2 - 0.01;  % 两个子图各占一半，减去间距
subWidth = 0.25;  % 子图宽度

z1Pos = [mainPos(1) + mainPos(3) + 0.03, mainPos(2) + mainPos(4)/2 + 0.005, subWidth, subHeight];
z2Pos = [mainPos(1) + mainPos(3) + 0.03, mainPos(2), subWidth, subHeight];

% 创建坐标轴
axMain = axes('Parent', fig, 'Position', mainPos);
axZ1 = axes('Parent', fig, 'Position', z1Pos);
axZ2 = axes('Parent', fig, 'Position', z2Pos);

%% 主图绘制
hs = surf(axMain, Xq, Yq, Zq, ...
    'FaceColor', 'interp', ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.98);

colormap(axMain, jet(256));  % 使用更多颜色级别

% Colorbar - 放在左边，垂直方向
cb = colorbar(axMain);
cb.Label.String = z_label;
cb.Position = cbarPos;  % 使用预设的左边位置
cb.FontSize = 10;

xlabel(axMain, x_label, 'FontSize', 12);
ylabel(axMain, y_label, 'FontSize', 12);
zlabel(axMain, z_label, 'FontSize', 12);

paperTitle = 'Thermocline surface over the study region (spatially continuous reconstruction)';
title(axMain, paperTitle, 'FontSize', 14, 'FontWeight', 'normal');

% 主图美化
axMain.XGrid = 'on'; axMain.YGrid = 'on'; axMain.ZGrid = 'on';
axMain.GridAlpha = 0.2;
axMain.Box = 'on';
axMain.FontName = 'Arial';
axMain.LineWidth = 1.0;
axMain.ZDir = 'reverse';

% 渲染设置
camproj(axMain, 'orthographic');
daspect(axMain, [1 1 0.35]);

xr = range(xKm);
yr = range(yKm);
ratio = xr / max(yr, eps);
ratio = min(max(ratio, 0.7), 1.4);
pbaspect(axMain, [ratio 1 0.55]);

switch viewPreset
    case "A"
        view(axMain, -135, 30);
    case "B"
        view(axMain, -135, 60);
    otherwise
        view(axMain, -135, 30);
end

% 主图光照
delete(findall(axMain, 'Type', 'light'));
material(axMain, 'dull');
lighting(axMain, 'gouraud');
axMain.AmbientLightColor = [0.35 0.35 0.35];
camlight(axMain, 'headlight');
camlight(axMain, 45, 30);

hs.SpecularStrength = 0.15;
hs.DiffuseStrength  = 0.85;
hs.AmbientStrength  = 0.35;

% 主图经纬度刻度
lonTicksDeg = linspace(min(lon), max(lon), nLonTicks);
latTicksDeg = linspace(min(lat), max(lat), nLatTicks);
xTicksKm = (lonTicksDeg - lon0) * km_per_deg_lon;
yTicksKm = (latTicksDeg - lat0) * km_per_deg_lat;

axMain.XTick = xTicksKm;
axMain.YTick = yTicksKm;
axMain.XTickLabel = compose('%.2f', lonTicksDeg);
axMain.YTickLabel = compose('%.2f', latTicksDeg);

xlim(axMain, [min(xKm) max(xKm)]);
ylim(axMain, [min(yKm) max(yKm)]);

%% 7) 选取两个放大区域
winDeg = [0.15, 0.10];   
minSepDeg = 0.15;        

densBinsLon = 20;  
densBinsLat = 20;

lonEdges = linspace(min(lon), max(lon), densBinsLon+1);
latEdges = linspace(min(lat), max(lat), densBinsLat+1);
N = histcounts2(lon, lat, lonEdges, latEdges);

lonCenters = 0.5 * (lonEdges(1:end-1) + lonEdges(2:end));
latCenters = 0.5 * (latEdges(1:end-1) + latEdges(2:end));

minPtsPerBin = 5;
N_filtered = N;
N_filtered(N < minPtsPerBin) = 0;

Nvec = N_filtered(:);
[~, order] = sort(Nvec, 'descend');

if isempty(order) || Nvec(order(1)) == 0
    c1 = [mean(lon), mean(lat)];
    c2 = [mean(lon)+(max(lon)-min(lon))*0.1, mean(lat)-(max(lat)-min(lat))*0.1];
else
    [i1, j1] = ind2sub(size(N_filtered), order(1));
    c1 = [lonCenters(i1), latCenters(j1)];

    c2 = [];
    for t = 2:numel(order)
        [ii, jj] = ind2sub(size(N_filtered), order(t));
        cand = [lonCenters(ii), latCenters(jj)];
        if Nvec(order(t)) == 0, break; end
        if hypot(cand(1)-c1(1), cand(2)-c1(2)) >= minSepDeg
            c2 = cand; break;
        end
    end
    if isempty(c2)
        [i2, j2] = ind2sub(size(N_filtered), order(2));
        c2 = [lonCenters(i2), latCenters(j2)];
    end
end

zoomCenters = [c1; c2];

%% 8) 绘制子图 - 传入主图坐标轴以同步颜色
% 获取全局Z轴范围用于颜色映射
zGlobalRange = [min(z(:)), max(z(:))];

drawZoom_final(axZ1, Xq, Yq, Zq, xKm, yKm, z, lon0, lat0, km_per_deg_lon, km_per_deg_lat, ...
           viewPreset, zoomCenters(1,:), winDeg, 1, axMain, zGlobalRange);
drawZoom_final(axZ2, Xq, Yq, Zq, xKm, yKm, z, lon0, lat0, km_per_deg_lon, km_per_deg_lat, ...
           viewPreset, zoomCenters(2,:), winDeg, 2, axMain, zGlobalRange);

% 主图上画方框
for k = 1:2
    cLon = zoomCenters(k,1);
    cLat = zoomCenters(k,2);

    lonMin = cLon - winDeg(1)/2; lonMax = cLon + winDeg(1)/2;
    latMin = cLat - winDeg(2)/2; latMax = cLat + winDeg(2)/2;

    xMin = (lonMin - lon0) * km_per_deg_lon; xMax = (lonMax - lon0) * km_per_deg_lon;
    yMin = (latMin - lat0) * km_per_deg_lat; yMax = (latMax - lat0) * km_per_deg_lat;

    hold(axMain, 'on');
    plot3(axMain, [xMin xMax xMax xMin xMin], [yMin yMin yMax yMax yMin], ...
        nan(1,5), 'k-', 'LineWidth', 1.2);
    hold(axMain, 'off');
end

%% 9) 导出（使用高DPI减少锯齿）
exportgraphics(fig, outname, 'Resolution', exportDPI);
fprintf('Figure saved: %s (Resolution=%d dpi)\n', outname, exportDPI);