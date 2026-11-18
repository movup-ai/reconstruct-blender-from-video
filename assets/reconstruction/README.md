# reconstruction/ — 场景数据与重建入口

| 文件 | 说明 |
| --- | --- |
| `scene.json` | 场景实际数据（schema `ba_whitemodel_scene` 1.0），由主工程保存后在同一 Blender 进程内导出（`pipeline/scene_export.py`） |
| `reconstruct.py` | 读取 / 重建入口：在 Blender 中按 `scene.json` 重建整个镜头并另存为新文件 |
| `README.md` | 本说明 |

## scene.json 内容

- **场景基准**：`scene` — 单位米、坐标系（Blender 右手系，+Z 向上，欧拉角 XYZ、弧度）、帧起止、帧率、画幅、活动相机、渲染设置（Workbench、色彩管理、世界颜色）。
- **对象和层级**：`collections`（集合及父集合）、`objects`（名称、类型、所属集合、父对象、静止变换、可见性、parent inverse 矩阵）。
- **几何和外观**：网格对象的全部顶点、面、每面材质槽、平滑标记；`materials`（材质名 → 颜色）。没有贴图。
- **角色和动作**：`characters` — 每个角色的标签、颜色、身高、根空物体、关节空物体、网格块。骨架为空物体层级（根：位置 + 朝向关键帧；骨盆：高度关键帧；各关节：XYZ 欧拉关键帧），每个网格块挂在对应关节下；白色眼部 / 胸前条带为材质 `WM_marker` 的面。没有约束和驱动。
- **动画**：`animation` — 所有关键帧（对象 → 属性 → 通道 → `[帧, 值, 插值]`），包括角色、相机、可见性切换和可动道具。
- **相机**：`camera` — 相机对象、焦距、传感器宽度、各镜头（shot）的帧段；相机每帧都有位置与旋转关键帧。
- 灯光：不适用（Workbench 工作室光照，无灯光对象）。

## 运行方法

适用软件：Blender（版本见 `scene.json` 的 `exporter` 字段），无需插件、无需网络、无其他依赖。

```
blender -b --factory-startup -P reconstruct.py -- scene.json rebuilt.blend
```

输出：`rebuilt.blend`（新文件，不会覆盖正式主工程 `model_3d/{包名}.blend`）。

## 已知限制

- 重建得到的是同样的对象、几何、材质颜色、层级与关键帧；Blender 用户界面布局等非场景数据不包含在内。
- 正式交付以主工程 `.blend` 为准；本文件和脚本仅用于接收方核对与重建。
