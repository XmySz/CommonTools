# 设置MOSQUITTO

### 1、安装[MOSQUITTO](https://mosquitto.org/download/)

### 2、在安装目录（通常为C:\Windows\Program Files\mosquitto）里找到配置文件mosquitto.conf ，任意位置添加下面三行：

```sql
allow_anonymous false
password_file C:\Windows\Program Files\mosquitto\pwfile.example
listener 1883 0.0.0.0
```

### 3、在mosquitto的安装目录下进入终端，运行下述命令：

```shell
mosquitto_passwd -c -b pwfile.example aa a123456
```

#### 4、在终端中运行下述命令运行代理服务器：

```
mosquitto -c mosquitto.conf -v
```



<hr style="height: 4px; background-color: black; border: none;">

# 设置MQTT

### 1、正常安装[MQTTX](https://mqttx.app/zh)

### 2、点击新建连接

### 3、按照下面进行配置，配置完成点击连接

- #### 名称：任意填写

- #### Client ID：确保不重复的前提下任意填写

- #### 服务器地址：写本地IP地址或者服务器IP

- #### 端口：1883

- #### 用户名：aa

- #### 密码：a123456

<hr style="height: 4px; background-color: black; border: none;">

## 编程工具

### Vscode/Pycharm+Python