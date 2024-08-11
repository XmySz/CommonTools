## <center>教程——给Ubuntu系统的某个通过命令行启动的应用添加图标</center>

### 一、创建.desktop文件

在~/.local/share/applications中创建这个应用的.desktop文件

### 二、编辑.desktop文件

```shell
[Desktop Entry]
Version=1.0
Name=My Application Name
Comment=My Application Description
Exec=/path/to/your/application
Icon=/path/to/your/icon.png
Terminal=false
Type=Application
Categories=Utility;
```

- Name：应用程序的名称。
- Exec：启动应用程序的命令或脚本的路径。
- Icon：应用程序图标的路径，需要是.png或.svg格式。
- Terminal：如果应用需要在终端中运行，则设置为true；否则为false。
- Categories：这个应用程序在菜单中的分类，如Utility、Development等。

### 三、添加可执行权限

```shell
chmod +x ~/.local/share/applications/myapp.desktop
```

