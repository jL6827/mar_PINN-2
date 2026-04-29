% 温跃层深度 3D 曲面图（散点插值 + 比例修正 + 轴显示经纬度 + 适量光照 + 更高分辨率） - MATLAB
% 改动：
% 1) 增加“适量光照”：双光源 + 材质 + 环境光（避免发黑）
% 2) 导出分辨率提高一档：300 -> 600 dpi (exportgraphics)

clear; clc;

%% 参数区
nx = 350;
ny = 350;

interpMethod = 'natural';   % 'natural' 或 'linear'
extrapMethod = 'none';      % 'none'（不外推）或 'nearest'（填满边界外）
doSmooth = true;
smoothWin = 7;

viewPreset = "A";           % "A": (-135,30)  "B": (-135,60)

outname = 'thermocline_Z0_3D_interp_lonlat_axes_light_600dpi.png';

% 轴刻度显示密度
nLonTicks = 6;
nLatTicks = 6;

% 导出分辨率（提高一档）
exportDPI = 600;

%% 1) 导入数据
data = load('thermocline_grid_data.mat');

Xg = data.Xg;
Yg = data.Yg;
Zg = data.Zg;

x_label   = data.x_label;   % Longitude
y_label   = data.y_label;   % Latitude
z_label   = data.z_label;   % Depth (m)
title_str = data.title_str;

%% 2) 散点化 + 去除 NaN/Inf
lon = Xg(:);
lat = Yg(:);
z   = Zg(:);

valid = isfinite(lon) & isfinite(lat) & isfinite(z);
lon = lon(valid); lat = lat(valid); z = z(valid);

% 去重 (lon,lat)
[ll_unique, ia] = unique([lon lat], 'rows', 'stable');
lon = ll_unique(:,1);
lat = ll_unique(:,2);
z   = z(ia);

%% 3) 经纬度 -> 近似 km 坐标（用于保持比例）
lon0 = mean(lon, 'omitnan');
lat0 = mean(lat, 'omitnan');

km_per_deg_lat = 111.32;
km_per_deg_lon = 111.32 * cosd(lat0);

xKm = (lon - lon0) * km_per_deg_lon;   % Easting (km)
yKm = (lat - lat0) * km_per_deg_lat;   % Northing (km)

%% 4) 插值器（在 km 坐标系下）
F = scatteredInterpolant(xKm, yKm, z, interpMethod, extrapMethod);

%% 5) 密网格 + 插值
xq = linspace(min(xKm), max(xKm), nx);
yq = linspace(min(yKm), max(yKm), ny);
[Xq, Yq] = meshgrid(xq, yq);

Zq = F(Xq, Yq);

if doSmooth
    Zq = smoothdata(Zq, 1, 'movmean', smoothWin, 'omitnan');
    Zq = smoothdata(Zq, 2, 'movmean', smoothWin, 'omitnan');
end

%% 6) 绘图
fig = figure('Position', [120, 80, 1200, 800], 'Color', 'w');
ax = axes('Parent', fig);

hs = surf(ax, Xq, Yq, Zq, ...
    'FaceColor', 'interp', ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.98);

colormap(ax, jet);
cb = colorbar(ax, 'Location', 'eastoutside');
cb.Label.String = z_label;

% 轴标签显示经纬度
xlabel(ax, x_label, 'FontSize', 12);
ylabel(ax, y_label, 'FontSize', 12);
zlabel(ax, z_label, 'FontSize', 12);
title(ax, title_str, 'FontSize', 14, 'FontWeight', 'normal');

ax.XGrid = 'on'; ax.YGrid = 'on'; ax.ZGrid = 'on';
ax.GridAlpha = 0.25;
ax.Box = 'on';
ax.FontName = 'Arial';
ax.LineWidth = 1.0;

% 深度轴方向（海洋习惯：向下更深）
ax.ZDir = 'reverse';

%% 7) 比例与投影（保持你满意的比例）
camproj(ax, 'orthographic');   % 正交投影减少透视变形（论文常用）
daspect(ax, [1 1 0.35]);       % Z 压缩（可微调）

xr = range(xKm);
yr = range(yKm);
ratio = xr / max(yr, eps);
ratio = min(max(ratio, 0.7), 1.4);
pbaspect(ax, [ratio 1 0.55]);

%% 8) 视角
switch viewPreset
    case "A"
        view(ax, -135, 30);
    case "B"
        view(ax, -135, 60);
    otherwise
        view(ax, -135, 30);
end

%% 9) 适量光照（避免“发黑”）
% 清理默认光（防止重复运行叠加光源）
delete(findall(fig, 'Type', 'light'));

% 材质与反射：让表面更亮但不过曝
material(ax, 'dull');          % dull/metal/shiny
lighting(ax, 'gouraud');       % 平滑光照

% 增加环境光（关键：整体提亮）
ax.AmbientLightColor = [0.35 0.35 0.35];  % 0~1，越大越亮

% 双光源：一个头灯 + 一个侧后补光，减少阴影死黑
camlight(ax, 'headlight');     % 随相机移动
camlight(ax,  45,  30);        % 固定方向补光（方位角/高度角）

% 若还偏暗，可把 Surface 的镜面反射稍微提一点点
hs.SpecularStrength = 0.15;    % 0~1
hs.DiffuseStrength  = 0.85;    % 0~1
hs.AmbientStrength  = 0.35;    % 0~1

%% 10) 把 km 轴刻度“显示为”经纬度刻度
lonTicksDeg = linspace(min(lon), max(lon), nLonTicks);
latTicksDeg = linspace(min(lat), max(lat), nLatTicks);

xTicksKm = (lonTicksDeg - lon0) * km_per_deg_lon;
yTicksKm = (latTicksDeg - lat0) * km_per_deg_lat;

ax.XTick = xTicksKm;
ax.YTick = yTicksKm;

ax.XTickLabel = compose('%.2f', lonTicksDeg);
ax.YTickLabel = compose('%.2f', latTicksDeg);
% 若你想带单位：
% ax.XTickLabel = compose('%.2f°E', lonTicksDeg);
% ax.YTickLabel = compose('%.2f°N', latTicksDeg);

xlim(ax, [min(xKm) max(xKm)]);
ylim(ax, [min(yKm) max(yKm)]);

%% 11) 导出更高分辨率
exportgraphics(fig, outname, 'Resolution', exportDPI);
fprintf('3D 图已保存：%s (Resolution=%d dpi)\n', outname, exportDPI);