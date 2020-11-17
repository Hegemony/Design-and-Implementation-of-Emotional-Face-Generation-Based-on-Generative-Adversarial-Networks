import sys
import os
import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QHBoxLayout, QLineEdit, QComboBox, \
    QFileDialog, QListWidget, QListView
from PyQt5.Qt import QFrame
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# from cirtorch.examples.retrieval import *


class EmotionAPP(QMainWindow):

    def __init__(self):
        super().__init__()
        self.cb_model = QComboBox(self)       # QComboBox 是下拉列表框组件类，它提供一个下拉列表供用户选择
        self.cb_dataset = QComboBox(self)
        self.lb_query_text = QLabel(self)     # 创建标签控件对象.参数：标签中要显示的文本
        self.lb_qimg_text = QLabel(self)
        self.lb_qimg = QLabel(self)
        self.label0_0 = QLabel(self)
        self.label0_1 = QLabel(self)
        self.label0_2 = QLabel(self)
        self.label0_3 = QLabel(self)
        self.label0_4 = QLabel(self)
        self.label0_5 = QLabel(self)
        self.label_0 = QLabel(self)
        self.label_1 = QLabel(self)
        self.label_2 = QLabel(self)
        self.label_3 = QLabel(self)
        self.label_4 = QLabel(self)
        self.label_5 = QLabel(self)
        self.label_6 = QLabel(self)
        self.label_7 = QLabel(self)
        self.label_8 = QLabel(self)
        self.label1_0 = QLabel(self)
        self.label1_1 = QLabel(self)
        self.label1_2 = QLabel(self)
        self.label1_3 = QLabel(self)
        self.label1_4 = QLabel(self)
        self.label1_5 = QLabel(self)
        self.label1_6 = QLabel(self)
        self.edt_query = QLineEdit(self)      # QLineEdit是一个单行文本输入框

        self.btn_select = QPushButton(self)   # QPushButton小部件提供了一个命令按钮。
        self.btn_search = QPushButton(self)   # 按钮或命令按钮可能是任何图形用户界面中最常用的小部件。

        # 模型和数据集相关参数
        self.model = None
        self.dataset = None
        self.lb_resultsimg = [QLabel(self) for i in range(20)]

        self.initUI()    # 调用下面定义的initUI()函数

    def initUI(self):

        # 模型和数据集选择下拉框
        self.cb_model.addItems(['模型', 'StarGAN', 'ganimation'])
        # 为QComboBox添加下拉表项，可以一次添加很多个
        self.cb_model.currentIndexChanged.connect(self.select_model)   # currentIndexChanged当下拉选项的索引发生改变时发射该信号,触发绑定的事件
        self.cb_model.move(800, 17)
        self.cb_dataset.addItems(['数据集', 'celeba', 'RaFD'])
        self.cb_dataset.currentIndexChanged.connect(self.select_dataset)
        self.cb_dataset.move(950, 17)

        # label
        self.lb_query_text.setText('Image path')
        self.lb_query_text.move(500, 72)
        self.edt_query.resize(500, self.lb_query_text.size().height())
        self.edt_query.move(620, 72)
        # self.lb_qimg_text.setText('query image')
        # self.lb_qimg_text.move(30, 92)
        cnt = 0
        x = 380
        y = 130
        for lb in self.lb_resultsimg:
            lb.setFixedSize(144, 192)
            # lb.setFrameStyle(QFrame.Panel)
            # lb.setText('result'+str(cnt))
            print(cnt)
            if cnt != 0 and cnt % 10 == 0:
                x = 380
                y += 195
            lb.move(x, y)
            x += 147
            cnt += 1


        # button
        self.btn_select.setText('Select')    # 设置按钮的文本
        self.btn_select.clicked.connect(self.select)  # 点击按钮触发操作
        self.btn_select.move(1150, 72)
        self.btn_search.setText('Generation')
        self.btn_search.clicked.connect(self.gengration_img)   # 点击按钮触发操作
        self.btn_search.move(1270, 72)

        # self.resize(600, 400)
        # self.setWindowTitle("label显示图片")

        # 更改人脸属性
        self.label0 = QLabel(self)
        self.label0.setText("   显示图片")
        self.label0.setFixedSize(1162, 128)
        self.label0.move(400, 200)

        self.label0.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                 )
        # 自定义一个标签文本框
        self.label0_0.setText('输入')
        self.label0_0.move(450, 172)
        self.label0_1.setText('黑发')
        self.label0_1.move(643, 172)
        self.label0_2.setText('金发')
        self.label0_2.move(836, 172)
        self.label0_3.setText('棕发')
        self.label0_3.move(1029, 172)
        self.label0_4.setText('男/女')
        self.label0_4.move(1222, 172)
        self.label0_5.setText('年轻/衰老')
        self.label0_5.move(1415, 172)

        btn0 = QPushButton(self)
        btn0.setText("更改人脸属性")
        btn0.move(850, 140)
        btn0.clicked.connect(self.openimage0)


        # 离散表情
        self.label = QLabel(self)
        self.label.setText("   显示图片")
        self.label.setFixedSize(1162, 128)
        self.label.move(400, 500)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                 )

        # 自定义一个标签文本框
        self.label_0.setText('输入')
        self.label_0.move(450, 472)
        self.label_1.setText('生气')
        self.label_1.move(578, 472)
        self.label_2.setText('轻蔑')
        self.label_2.move(706, 472)
        self.label_3.setText('厌恶')
        self.label_3.move(834, 472)
        self.label_4.setText('恐惧')
        self.label_4.move(962, 472)
        self.label_5.setText('快乐')
        self.label_5.move(1090, 472)
        self.label_6.setText('中立')
        self.label_6.move(1218, 472)
        self.label_7.setText('伤心')
        self.label_7.move(1346, 472)
        self.label_8.setText('惊讶')
        self.label_8.move(1474, 472)

        btn = QPushButton(self)
        btn.setText("离散表情生成")
        btn.move(850, 440)
        btn.clicked.connect(self.openimage)

        # 连续表情
        self.label1 = QLabel(self)
        self.label1.setText("   显示图片")
        self.label1.setFixedSize(1162, 128)
        self.label1.move(400, 800)

        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                 )

        # 自定义一个标签文本框
        self.label1_0.setText('输入')
        self.label1_0.move(450, 772)
        self.label1_1.setText('20%')
        self.label1_1.move(616, 772)
        self.label1_2.setText('40%')
        self.label1_2.move(782, 772)
        self.label1_3.setText('60%')
        self.label1_3.move(948, 772)
        self.label1_4.setText('80%')
        self.label1_4.move(1114, 772)
        self.label1_5.setText('100%')
        self.label1_5.move(1280, 772)
        self.label1_6.setText('目标域')
        self.label1_6.move(1446, 772)

        # 关闭窗口提示
        btn1 = QPushButton(self)
        btn1.setText("连续表情生成")
        btn1.move(850, 740)
        btn1.clicked.connect(self.openimage1)



        # 设置窗口位置和大小
        # self.setGeometry(200, 200, 1000, 800)
        # 设置窗口标题
        self.setWindowTitle('Emotional Face Generation')
        self.setWindowIcon(QIcon('web.png'))

        self.showMaximized()

    def openimage0(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "stargan_celeba/", "*.jpg;;*.png;;*.gif;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label0.setPixmap(jpg)

    def openimage(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "stargan_rafd/", "*.jpg;;*.png;;*.gif;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)


    def openimage1(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "results/", "*.jpg;;*.png;;*.gif;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label1.setPixmap(jpg)


    def select(self):   # 选择数据集位置，显示到前端
        download_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                   "浏览",
                                                                   "data0/")
        self.edt_query.setText(download_path)
        print(download_path)


# def select_query_image(self):
#         if self.dataset is None:
#             dir_query = 'query_images'
#         else:
#             dir_query = os.path.join('query_images', self.dataset)
#         # query_image_path = QFileDialog.getOpenFileName(self, "浏览", dir_query, "JPEG Files(*.jpg)")
#         query_image_path = QFileDialog.getOpenFileNames(self, "浏览", dir_query, "JPEG Files(*.jpg)")
#         # 打开文件有以下3种：
#         # 1、单个文件打开 QFileDialog.getOpenFileName()
#         # 2、多个文件打开 QFileDialog.getOpenFileNames()
#         # 3、打开文件夹 QFileDialog.getExistingDirectory()
#         self.edt_query.setText(query_image_path[0])
#         img = QImage(query_image_path[0])
#         img = img.scaled(img.width() * 0.3, img.height() * 0.3)
#         self.lb_qimg.resize(img.width(), img.height())
#         self.lb_qimg.setPixmap(QPixmap.fromImage(img))
#         self.lb_qimg.move(30, 140)

    def select_model(self):
        model_name = self.cb_model.currentText().lower()   # 返回选中选项的文本
        if model_name == '模型':
            return
        self.model = model_name
        print(self.model)

    def select_dataset(self):
        self.dataset = self.cb_dataset.currentText().lower()
        if self.dataset == '数据集':
            return
        print(self.dataset)


    def closeEvent(self, event):  # 我们关闭窗口的时候,触发了QCloseEvent。我们需要重写closeEvent()事件处理程序。

        reply = QMessageBox.question(self, 'QUIT',
                                         "Are you sure to quit?", QMessageBox.Yes |
                                         QMessageBox.No, QMessageBox.No)
        # 我们显示一个消息框,两个按钮:“是”和“不是”。第一个字符串出现在titlebar。第二个字符串消息对话框中显示的文本。第三个参数指定按钮的组合出现在对话框中。最后一个参数是默认按钮，这个是默认的按钮焦点。
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def gengration_img(self):

        input = self.edt_query.text()
        print(input)
        # results = retrieval(self.model, self.dataset, input)  # 原始代码

        # for i in range(len(results)):
        #     img = QImage(results[i])
        #     self.lb_resultsimg[i].setPixmap(QPixmap.fromImage(img))
        #     self.lb_resultsimg[i].setScaledContents(True)
        if self.model == 'stargan' and self.dataset == 'celeba':
            os.system('python main0.py --mode test --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young')
        elif self.model == 'stargan' and self.dataset == 'rafd':
            os.system('python main0.py --mode test --dataset RaFD --image_size 128 --c_dim 8 --rafd_image_dir data0/RaFD/test --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results')
        elif self.model == 'ganimation' and self.dataset == 'celeba':
            os.system('python main.py --mode test --data_root datasets/celebA --batch_size 8 --max_dataset_size 150 --gpu_ids 0 --ckpt_dir ckpts/celebA/ganimation/190327_161852/ --load_epoch 30')
        elif self.model == 'ganimation' and self.dataset == 'rafd':
            os.system('python main.py --mode test --data_root datasets/rafd --batch_size 8 --max_dataset_size 150 --gpu_ids 0 --ckpt_dir ckpts/rafd/ganimation/200418_165658/ --load_epoch 300')

if __name__ == '__main__':
    # 创建应用程序对象
    app = QApplication(sys.argv)
    ex = EmotionAPP()
    sys.exit(app.exec_())