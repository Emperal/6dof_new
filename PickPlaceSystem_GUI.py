
import os
import sys
import time
import traceback
import importlib.util
import threading
import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QFileDialog, QSpinBox, QMessageBox, QLineEdit
)


# ==========================================================
# 工具：将 stdout / stderr 重定向到 Qt 日志框
# ==========================================================
class EmittingStream(QObject):
    text_written = Signal(str)

    def write(self, text):
        if text:
            self.text_written.emit(str(text))

    def flush(self):
        pass


# ==========================================================
# 工具：OpenCV BGR -> QPixmap
# ==========================================================
def cvimg_to_qpixmap(img_bgr):
    if img_bgr is None:
        return QPixmap()

    if len(img_bgr.shape) == 2:
        h, w = img_bgr.shape
        qimg = QImage(img_bgr.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ==========================================================
# 加载第二个程序核心
# ==========================================================
def load_core_module(py_file_path):
    if not os.path.exists(py_file_path):
        raise FileNotFoundError(f"核心程序不存在: {py_file_path}")

    module_name = "pick_place_core_module"
    spec = importlib.util.spec_from_file_location(module_name, py_file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载核心程序: {py_file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================
# 初始化线程：避免 GUI 主线程卡死
# ==========================================================
class InitWorker(QThread):
    ok = Signal(object, object)   # core, system
    failed = Signal(str)
    log_signal = Signal(str)

    def __init__(self, core_file, mesh_file, template_dir, save_root):
        super().__init__()
        self.core_file = core_file
        self.mesh_file = mesh_file
        self.template_dir = template_dir
        self.save_root = save_root

    def run(self):
        try:
            self.log_signal.emit("正在加载程序核心...")
            core = load_core_module(self.core_file)

            # 重置第二个程序中的全局状态
            core.current_pick_result = None

            try:
                core.stop_event.clear()
            except Exception:
                core.stop_event = threading.Event()

            try:
                core.vision_request_event.clear()
            except Exception:
                core.vision_request_event = threading.Event()

            try:
                core.set_latest_vis(None)
            except Exception:
                pass

            self.log_signal.emit("正在初始化系统...")
            system = core.PickPlaceSystem(
                mesh_file=self.mesh_file,
                template_dir=self.template_dir,
                save_root=self.save_root
            )

            self.ok.emit(core, system)

        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ==========================================================
# 预览线程：相机读帧放到后台，避免阻塞 GUI
# ==========================================================
class PreviewWorker(QThread):
    frame_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, system, fps=15):
        super().__init__()
        self.system = system
        self.fps = max(1, int(fps))
        self._running = True
        self._last_error = ""
        self._last_error_ts = 0.0

    def stop(self):
        self._running = False
        self.requestInterruption()
        self.wait(1500)

    def run(self):
        interval_ms = max(1, int(1000 / self.fps))

        while self._running and not self.isInterruptionRequested():
            try:
                frame, _ = self.system.get_current_frame()
                self.frame_ready.emit(frame)
            except Exception as e:
                err = f"预览线程读帧失败: {e}"
                now = time.time()
                if err != self._last_error or (now - self._last_error_ts) > 2.0:
                    self._last_error = err
                    self._last_error_ts = now
                    self.failed.emit(err)
                self.msleep(200)
                continue

            self.msleep(interval_ms)


# ==========================================================
# 单次执行线程
# ==========================================================
class SingleCycleWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, core_module, system, refine_iter):
        super().__init__()
        self.core = core_module
        self.system = system
        self.refine_iter = refine_iter

    def run(self):
        try:
            self.system.move_robot_photo_pose()

            ret = self.system.estimate_pick_once(refine_iter=self.refine_iter)
            if ret["status"] != "ok":
                self.finished_ok.emit("单次执行结束：识别失败或被过滤")
                return

            pick_data = ret["data"]
            self.system.execute_pick(pick_data["pick_pose6"])
            self.system.execute_place()

            self.finished_ok.emit("单次任务执行完成")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ==========================================================
# 自动循环线程：直接调用第二个程序的 pick_place_cycle(system)
# ==========================================================
class AutoCycleWorker(QThread):
    log_signal = Signal(str)
    failed = Signal(str)
    finished_ok = Signal(str)

    def __init__(self, core_module, system):
        super().__init__()
        self.core = core_module
        self.system = system

    def run(self):
        try:
            self.log_signal.emit("自动循环已启动")
            self.core.pick_place_cycle(self.system)
            self.finished_ok.emit("自动循环已退出")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ==========================================================
# 回 Home 线程：避免机器人移动阻塞 GUI
# ==========================================================
class HomeWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, system):
        super().__init__()
        self.system = system

    def run(self):
        try:
            self.system.move_robot_photo_pose()
            self.finished_ok.emit("机器人已执行回 Home")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# ==========================================================
# 主界面
# ==========================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("抓取放置系统 GUI")
        self.resize(1450, 860)

        self.core = None
        self.system = None

        self.init_worker = None
        self.preview_worker = None
        self.single_worker = None
        self.auto_worker = None
        self.home_worker = None
        self.vision_thread = None

        self._build_ui()

        # stdout / stderr 重定向
        self.stdout_stream = EmittingStream()
        self.stderr_stream = EmittingStream()
        self.stdout_stream.text_written.connect(self.append_log_text)
        self.stderr_stream.text_written.connect(self.append_log_text)

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream

        # 定时刷新识别结果窗口（这个动作很轻，保留定时器即可）
        self.result_timer = QTimer(self)
        self.result_timer.timeout.connect(self.update_result_view)
        self.result_timer.start(100)

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------
    def _build_ui(self):
        param_group = QGroupBox("参数设置")

        self.core_edit = QLineEdit()
        self.mesh_edit = QLineEdit()
        self.template_edit = QLineEdit()
        self.save_root_edit = QLineEdit()

        self.refine_spin = QSpinBox()
        self.refine_spin.setRange(1, 50)
        self.refine_spin.setValue(5)

        code_dir = os.path.dirname(os.path.abspath(__file__))
        self.core_edit.setText(os.path.join(code_dir, "PickPlaceSystem.py"))
        self.mesh_edit.setText(os.path.join(code_dir, "demo_data_pian/my_data0/mesh/pian_hole_m.obj"))
        self.template_edit.setText(os.path.join(code_dir, "templates1280*720/back"))
        self.save_root_edit.setText(os.path.join(code_dir, "demo_data_pian/PickPlaceSystem_output"))

        btn_core = QPushButton("选择核心程序")
        btn_mesh = QPushButton("选择 mesh")
        btn_template = QPushButton("选择模板目录")
        btn_save_root = QPushButton("选择输出目录")

        btn_core.clicked.connect(self.select_core_file)
        btn_mesh.clicked.connect(self.select_mesh)
        btn_template.clicked.connect(self.select_template_dir)
        btn_save_root.clicked.connect(self.select_save_root)

        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("core_file:"), 0, 0)
        param_layout.addWidget(self.core_edit, 0, 1)
        param_layout.addWidget(btn_core, 0, 2)

        param_layout.addWidget(QLabel("mesh_file:"), 1, 0)
        param_layout.addWidget(self.mesh_edit, 1, 1)
        param_layout.addWidget(btn_mesh, 1, 2)

        param_layout.addWidget(QLabel("template_dir:"), 2, 0)
        param_layout.addWidget(self.template_edit, 2, 1)
        param_layout.addWidget(btn_template, 2, 2)

        param_layout.addWidget(QLabel("save_root:"), 3, 0)
        param_layout.addWidget(self.save_root_edit, 3, 1)
        param_layout.addWidget(btn_save_root, 3, 2)

        param_layout.addWidget(QLabel("refine_iter:"), 4, 0)
        param_layout.addWidget(self.refine_spin, 4, 1)

        param_group.setLayout(param_layout)

        control_group = QGroupBox("控制")

        self.btn_init = QPushButton("初始化系统")
        self.btn_run_once = QPushButton("单次执行")
        self.btn_start_auto = QPushButton("开始自动循环")
        self.btn_stop_auto = QPushButton("停止自动循环")
        self.btn_home = QPushButton("回 Home")
        self.btn_exit = QPushButton("退出")

        self.btn_run_once.setEnabled(False)
        self.btn_start_auto.setEnabled(False)
        self.btn_stop_auto.setEnabled(False)
        self.btn_home.setEnabled(False)

        self.btn_init.clicked.connect(self.init_system)
        self.btn_run_once.clicked.connect(self.run_once)
        self.btn_start_auto.clicked.connect(self.start_auto)
        self.btn_stop_auto.clicked.connect(self.stop_auto)
        self.btn_home.clicked.connect(self.go_home)
        self.btn_exit.clicked.connect(self.close)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_init)
        control_layout.addWidget(self.btn_run_once)
        control_layout.addWidget(self.btn_start_auto)
        control_layout.addWidget(self.btn_stop_auto)
        control_layout.addWidget(self.btn_home)
        control_layout.addWidget(self.btn_exit)

        control_group.setLayout(control_layout)

        image_group = QGroupBox("图像显示")

        self.preview_label = QLabel("相机预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background-color: black; color: white;")

        self.result_label = QLabel("识别结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(640, 480)
        self.result_label.setStyleSheet("background-color: black; color: white;")

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.preview_label, 1)
        img_layout.addWidget(self.result_label, 1)
        image_group.setLayout(img_layout)

        log_group = QGroupBox("日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(param_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(image_group, 1)
        main_layout.addWidget(log_group, 1)

        self.setLayout(main_layout)

    # ------------------------------------------------------
    # 日志与显示
    # ------------------------------------------------------
    def append_log_text(self, text):
        if not text:
            return
        text = text.rstrip("\n")
        if not text:
            return
        t = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{t}] {text}")

    def log(self, text):
        t = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{t}] {text}")

    def show_blank_preview(self, text="相机预览"):
        self.preview_label.setText(text)
        self.preview_label.setPixmap(QPixmap())

    def show_blank_result(self, text="识别结果"):
        self.result_label.setText(text)
        self.result_label.setPixmap(QPixmap())

    def on_preview_frame(self, frame):
        try:
            pix = cvimg_to_qpixmap(frame)
            self.preview_label.setPixmap(
                pix.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        except Exception as e:
            self.log(f"预览显示失败: {e}")

    def update_result_view(self):
        if self.core is None:
            return

        try:
            vis = self.core.get_latest_vis()
            if vis is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                vis = blank

            pix = cvimg_to_qpixmap(vis)
            self.result_label.setPixmap(
                pix.scaled(
                    self.result_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        except Exception as e:
            self.log(f"结果窗口刷新失败: {e}")

    # ------------------------------------------------------
    # 文件选择
    # ------------------------------------------------------
    def select_core_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择第二个程序核心文件", "", "Python Files (*.py)")
        if path:
            self.core_edit.setText(path)

    def select_mesh(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 mesh 文件", "", "OBJ Files (*.obj);;All Files (*)")
        if path:
            self.mesh_edit.setText(path)

    def select_template_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择模板目录")
        if path:
            self.template_edit.setText(path)

    def select_save_root(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.save_root_edit.setText(path)

    # ------------------------------------------------------
    # 状态判断
    # ------------------------------------------------------
    def is_init_running(self):
        return self.init_worker is not None and self.init_worker.isRunning()

    def is_single_running(self):
        return self.single_worker is not None and self.single_worker.isRunning()

    def is_auto_running(self):
        return self.auto_worker is not None and self.auto_worker.isRunning()

    def is_home_running(self):
        return self.home_worker is not None and self.home_worker.isRunning()

    def any_busy_action(self):
        return self.is_init_running() or self.is_single_running() or self.is_auto_running() or self.is_home_running()

    def refresh_button_states(self):
        has_system = (self.system is not None and self.core is not None)
        busy_init = self.is_init_running()
        busy_single = self.is_single_running()
        busy_auto = self.is_auto_running()
        busy_home = self.is_home_running()

        self.btn_init.setEnabled(not busy_init and not busy_single and not busy_home)

        self.btn_run_once.setEnabled(has_system and (not busy_init) and (not busy_single) and (not busy_auto) and (not busy_home))
        self.btn_start_auto.setEnabled(has_system and (not busy_init) and (not busy_single) and (not busy_auto) and (not busy_home))
        self.btn_stop_auto.setEnabled(has_system and (busy_auto or (self.vision_thread is not None and self.vision_thread.is_alive())))
        self.btn_home.setEnabled(has_system and (not busy_init) and (not busy_single) and (not busy_auto) and (not busy_home))

    # ------------------------------------------------------
    # 线程管理
    # ------------------------------------------------------
    def start_preview_worker(self):
        self.stop_preview_worker()

        if self.system is None:
            return

        self.preview_worker = PreviewWorker(self.system, fps=15)
        self.preview_worker.frame_ready.connect(self.on_preview_frame)
        self.preview_worker.failed.connect(self.log)
        self.preview_worker.start()

    def stop_preview_worker(self):
        if self.preview_worker is not None:
            try:
                self.preview_worker.stop()
            except Exception:
                pass
            self.preview_worker = None

    def stop_auto(self):
        was_running = self.is_auto_running() or (self.vision_thread is not None and self.vision_thread.is_alive())

        if self.core is not None:
            try:
                self.core.stop_event.set()
                self.core.vision_request_event.set()
                self.core.current_pick_result = None
            except Exception:
                pass

        if self.auto_worker is not None and self.auto_worker.isRunning():
            self.auto_worker.wait(3000)

        if self.vision_thread is not None and self.vision_thread.is_alive():
            try:
                self.vision_thread.join(timeout=1.5)
            except Exception:
                pass

        self.auto_worker = None
        self.vision_thread = None

        if was_running:
            self.log("自动循环已停止")

        self.refresh_button_states()

    def shutdown_system_only(self):
        self.stop_auto()
        self.stop_preview_worker()

        try:
            if self.system is not None and hasattr(self.system, "camera") and hasattr(self.system.camera, "release"):
                self.system.camera.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ------------------------------------------------------
    # 初始化
    # ------------------------------------------------------
    def init_system(self):
        if self.is_init_running():
            self.log("初始化正在进行中")
            return

        if self.is_single_running() or self.is_home_running():
            QMessageBox.warning(self, "提示", "当前有任务在执行，请等待完成后再初始化")
            return

        if self.is_auto_running():
            self.stop_auto()

        if self.system is not None:
            try:
                self.shutdown_system_only()
            except Exception:
                pass
            self.system = None
            self.core = None

        core_file = self.core_edit.text().strip()
        mesh_file = self.mesh_edit.text().strip()
        template_dir = self.template_edit.text().strip()
        save_root = self.save_root_edit.text().strip()

        if not os.path.exists(core_file):
            QMessageBox.warning(self, "错误", f"核心程序不存在：\n{core_file}")
            return
        if not os.path.exists(mesh_file):
            QMessageBox.warning(self, "错误", f"mesh 文件不存在：\n{mesh_file}")
            return
        if not os.path.exists(template_dir):
            QMessageBox.warning(self, "错误", f"模板目录不存在：\n{template_dir}")
            return

        os.makedirs(save_root, exist_ok=True)

        self.show_blank_preview("相机预览")
        self.show_blank_result("识别结果")

        self.init_worker = InitWorker(core_file, mesh_file, template_dir, save_root)
        self.init_worker.log_signal.connect(self.log)
        self.init_worker.ok.connect(self.on_init_ok)
        self.init_worker.failed.connect(self.on_init_failed)
        self.init_worker.finished.connect(self.refresh_button_states)
        self.init_worker.start()

        self.refresh_button_states()

    def on_init_ok(self, core, system):
        self.core = core
        self.system = system

        self.start_preview_worker()
        self.log("系统初始化成功")
        self.refresh_button_states()

    def on_init_failed(self, err):
        self.system = None
        self.core = None
        QMessageBox.critical(self, "初始化失败", err)
        self.log(f"初始化失败: {err}")
        self.refresh_button_states()

    # ------------------------------------------------------
    # 单次执行
    # ------------------------------------------------------
    def run_once(self):
        if self.system is None or self.core is None:
            QMessageBox.warning(self, "提示", "请先初始化系统")
            return

        if self.is_single_running():
            self.log("已有单次任务正在执行")
            return

        if self.is_auto_running():
            self.log("自动循环运行中，不能执行单次任务")
            return

        if self.is_home_running():
            self.log("回 Home 正在执行，不能启动单次任务")
            return

        self.single_worker = SingleCycleWorker(
            self.core,
            self.system,
            self.refine_spin.value()
        )
        self.single_worker.finished_ok.connect(self.on_single_finished)
        self.single_worker.failed.connect(self.on_worker_failed)
        self.single_worker.finished.connect(self.refresh_button_states)
        self.single_worker.start()

        self.log("开始执行单次任务")
        self.refresh_button_states()

    def on_single_finished(self, msg):
        self.log(msg)

    # ------------------------------------------------------
    # 自动循环
    # ------------------------------------------------------
    def start_auto(self):
        if self.system is None or self.core is None:
            QMessageBox.warning(self, "提示", "请先初始化系统")
            return

        if self.is_auto_running():
            self.log("自动循环已经在运行")
            return

        if self.is_single_running():
            self.log("单次任务运行中，不能启动自动循环")
            return

        if self.is_home_running():
            self.log("回 Home 正在执行，不能启动自动循环")
            return

        try:
            self.core.current_pick_result = None
        except Exception:
            pass

        try:
            self.core.stop_event.clear()
        except Exception:
            self.core.stop_event = threading.Event()

        try:
            self.core.vision_request_event.clear()
        except Exception:
            self.core.vision_request_event = threading.Event()

        # 每次启动自动循环时，都重新启动第二个程序的异步视觉线程
        self.vision_thread = threading.Thread(
            target=self.core.vision_worker,
            args=(self.system, self.refine_spin.value()),
            daemon=True
        )
        self.vision_thread.start()

        self.auto_worker = AutoCycleWorker(self.core, self.system)
        self.auto_worker.log_signal.connect(self.log)
        self.auto_worker.finished_ok.connect(self.log)
        self.auto_worker.failed.connect(self.on_worker_failed)
        self.auto_worker.finished.connect(self.refresh_button_states)
        self.auto_worker.start()

        self.refresh_button_states()

    # ------------------------------------------------------
    # 回 Home
    # ------------------------------------------------------
    def go_home(self):
        if self.system is None:
            QMessageBox.warning(self, "提示", "请先初始化系统")
            return

        if self.is_single_running():
            self.log("单次任务运行中，不能执行回 Home")
            return

        if self.is_auto_running():
            self.log("自动循环运行中，不能执行回 Home")
            return

        if self.is_home_running():
            self.log("回 Home 已在执行")
            return

        self.home_worker = HomeWorker(self.system)
        self.home_worker.finished_ok.connect(self.on_home_finished)
        self.home_worker.failed.connect(self.on_worker_failed)
        self.home_worker.finished.connect(self.refresh_button_states)
        self.home_worker.start()

        self.log("开始执行回 Home")
        self.refresh_button_states()

    def on_home_finished(self, msg):
        self.log(msg)

    # ------------------------------------------------------
    # 通用错误处理
    # ------------------------------------------------------
    def on_worker_failed(self, err):
        self.log(f"任务失败: {err}")
        QMessageBox.critical(self, "任务失败", err)
        self.refresh_button_states()

    # ------------------------------------------------------
    # 关闭
    # ------------------------------------------------------
    def closeEvent(self, event):
        try:
            self.result_timer.stop()
        except Exception:
            pass

        try:
            if self.init_worker is not None and self.init_worker.isRunning():
                self.init_worker.wait(2000)
        except Exception:
            pass

        try:
            if self.single_worker is not None and self.single_worker.isRunning():
                self.single_worker.wait(2000)
        except Exception:
            pass

        try:
            if self.home_worker is not None and self.home_worker.isRunning():
                self.home_worker.wait(2000)
        except Exception:
            pass

        try:
            self.shutdown_system_only()
        except Exception:
            pass

        try:
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()