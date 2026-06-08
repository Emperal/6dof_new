import socket, time

class SocketClient:
    def __init__(self, ip="192.168.58.4", port=2000, timeout=3.0, keep_alive=True):
        self.ip = ip
        self.port = port
        self.keep_alive = keep_alive
        self.sock = None
        self.timeout = timeout
        self.connect()

    def connect(self):
        """建立连接（带重试）"""
        for i in range(3):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.ip, self.port))
                print(f"[✅ 已连接] {self.ip}:{self.port}")
                return
            except Exception as e:
                print(f"[⚠️ 连接失败 第{i+1}次] {e}")
                time.sleep(0.5)
        print("[❌ 无法连接服务器]")
        self.sock = None

    def send(self, msg: str):
        """发送 '0'/'1'，自动检测掉线重连"""
        if msg not in ("0", "1","2","3"):
            raise ValueError("只能发送'0''1''2''3'")
        if not self.sock:
            print("[🔁 尝试重连中...]")
            self.connect()
            if not self.sock:
                print("[❌ 发送失败：无法重连]")
                return
        try:
            self.sock.sendall(msg.encode("utf-8"))
            print(f"[📤 已发送] {msg}")
            if not self.keep_alive:
                self.close()
        except Exception as e:
            print(f"[⚠️ 发送失败] {e}")
            self.close()
            if self.keep_alive:
                print("[🔁 自动重连中...]")
                self.connect()

    def close(self):
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.sock.close()
            self.sock = None
            print("[🔌 已关闭连接]")

# ========== 测试入口 ==========
if __name__ == "__main__":
    client = SocketClient("192.168.58.4", 2000)

    client.send("1")

    # client.send("1")
    client.close()
